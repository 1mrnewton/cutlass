//! Korean written letter by letter.
//!
//! A Hangul syllable is one character, or the two or three letters it is
//! made of: a leading consonant, a vowel, and sometimes a trailing
//! consonant. Text that has been taken apart, as a Mac's file names are,
//! comes as the letters. They are put back into the syllable when the
//! face has it; a syllable the face lacks is taken apart when the face
//! has its letters; and letters that stay apart are marked so the face's
//! own features can join them, which is how old Hangul is drawn.

use cutlass_font::Font;

use super::buffer::Glyph;
use super::{Feature, has};

/// The bits of a glyph's mask for a leading consonant, a vowel and a
/// trailing consonant that stand apart.
pub(super) const LEADING: u32 = 1 << 9;
pub(super) const VOWEL: u32 = 1 << 10;
pub(super) const TRAILING: u32 = 1 << 11;

const SYLLABLES: u32 = 0xAC00;
const LEADS: u32 = 0x1100;
const VOWELS: u32 = 0x1161;
/// One before the first trailing consonant: a syllable with none counts 0.
const TRAILS: u32 = 0x11A7;
/// Syllables to each leading consonant, and to each vowel.
const PER_LEAD: u32 = 588;
const PER_VOWEL: u32 = 28;
const SYLLABLE_COUNT: u32 = 11_172;

fn leading(code: u32) -> bool {
    matches!(code, 0x1100..=0x115F | 0xA960..=0xA97C)
}

fn vowel(code: u32) -> bool {
    matches!(code, 0x1160..=0x11A7 | 0xD7B0..=0xD7C6)
}

fn trailing(code: u32) -> bool {
    matches!(code, 0x11A8..=0x11FF | 0xD7CB..=0xD7FB)
}

/// The letters a modern syllable is made of.
fn modern_lead(code: u32) -> bool {
    (LEADS..LEADS + 19).contains(&code)
}

fn modern_vowel(code: u32) -> bool {
    (VOWELS..VOWELS + 21).contains(&code)
}

fn modern_trail(code: u32) -> bool {
    (TRAILS + 1..TRAILS + PER_VOWEL).contains(&code)
}

fn syllable(code: u32) -> bool {
    (SYLLABLES..SYLLABLES + SYLLABLE_COUNT).contains(&code)
}

/// The two marks of pitch old Korean writes after a syllable and draws at
/// its left.
fn tone(code: u32) -> bool {
    matches!(code, 0x302E | 0x302F)
}

/// The letters that stand for a missing consonant or vowel. Unicode lets
/// a renderer leave them out; Korean faces draw them, most as nothing
/// with no room, and their joins expect them there.
pub(super) fn filler(character: char) -> bool {
    matches!(character, '\u{115F}' | '\u{1160}' | '\u{3164}' | '\u{FFA0}')
}

const DOTTED_CIRCLE: char = '\u{25CC}';

/// The features that join letters standing apart, added to the `stages`
/// of a Korean run. Contextual alternates are left out: the faces that
/// hold every one of these joins under that feature would apply them to
/// plain syllables too.
pub(super) fn stages(stages: &mut [Vec<Feature>]) {
    for stage in stages.iter_mut() {
        stage.retain(|feature| &feature.tag != b"calt");
    }
    if let Some(last) = stages.last_mut() {
        last.extend([
            Feature::masked(b"ljmo", LEADING),
            Feature::masked(b"vjmo", VOWEL),
            Feature::masked(b"tjmo", TRAILING),
        ]);
    }
}

/// Put the letters of `glyphs` into syllables the face has, and take
/// apart the syllables it lacks. A mark of pitch after a syllable goes
/// before it, where it is drawn, unless the face draws it with no room of
/// its own; one after no syllable stands by a dotted circle when the face
/// has one. `describe` makes the glyph of a character at a place in the
/// text.
pub(super) fn compose(
    font: &Font<'_>,
    glyphs: Vec<Glyph>,
    describe: impl Fn(char, usize) -> Glyph,
) -> Vec<Glyph> {
    let code = |glyph: &Glyph| u32::from(glyph.character);
    let has_code = |code: u32| char::from_u32(code).is_some_and(|character| has(font, character));
    // A mark the face lacks is drawn as its missing glyph, which has room.
    let roomless = |character: char| {
        font.glyph_id(character)
            .ok()
            .flatten()
            .is_some_and(|id| font.advance(id).is_ok_and(|advance| advance == 0))
    };
    let mut out: Vec<Glyph> = Vec::with_capacity(glyphs.len());
    // Where the syllable just written starts in `out`, while nothing
    // has followed it.
    let mut syllable_start: Option<usize> = None;
    let mut at = 0;
    while at < glyphs.len() {
        let first = code(&glyphs[at]);
        let cluster = glyphs[at].cluster;
        let next = glyphs.get(at + 1).map(code);
        if tone(first) {
            let mut mark = glyphs[at].clone();
            let in_place = roomless(mark.character);
            match syllable_start.take() {
                Some(start) if !in_place => {
                    // The mark and its syllable are one cluster.
                    let cluster = out[start..]
                        .iter()
                        .map(|glyph| glyph.cluster)
                        .fold(mark.cluster, usize::min);
                    for glyph in &mut out[start..] {
                        glyph.cluster = cluster;
                        glyph.continues = true;
                    }
                    mark.cluster = cluster;
                    mark.continues = false;
                    out.insert(start, mark);
                }
                Some(_) => out.push(mark),
                None if has(font, DOTTED_CIRCLE) => {
                    let mut circle = describe(DOTTED_CIRCLE, cluster);
                    if in_place {
                        mark.continues = true;
                        out.extend([circle, mark]);
                    } else {
                        circle.continues = true;
                        mark.continues = false;
                        out.extend([mark, circle]);
                    }
                }
                None => out.push(mark),
            }
            at += 1;
            continue;
        }
        let before = out.len();
        if leading(first) && next.is_some_and(vowel) {
            let middle = next.unwrap_or(0);
            let last = glyphs.get(at + 2).map(code).filter(|last| trailing(*last));
            let length = if last.is_some() { 3 } else { 2 };
            // One syllable of modern letters the face has is one glyph.
            if modern_lead(first) && modern_vowel(middle) && last.is_none_or(modern_trail) {
                let whole = SYLLABLES
                    + (first - LEADS) * PER_LEAD
                    + (middle - VOWELS) * PER_VOWEL
                    + last.map_or(0, |last| last - TRAILS);
                if has_code(whole)
                    && let Some(character) = char::from_u32(whole)
                {
                    out.push(describe(character, cluster));
                    at += length;
                    syllable_start = Some(before);
                    continue;
                }
            }
            // The letters stay apart, each marked for the face's joins.
            for (offset, mask) in [LEADING, VOWEL, TRAILING]
                .into_iter()
                .take(length)
                .enumerate()
            {
                let mut letter = glyphs[at + offset].clone();
                letter.mask |= mask;
                letter.cluster = cluster;
                letter.continues = offset > 0;
                out.push(letter);
            }
            at += length;
            syllable_start = Some(before);
            continue;
        }
        if syllable(first) {
            let index = first - SYLLABLES;
            let (lead, rest) = (index / PER_LEAD, index % PER_LEAD);
            let (middle, trail) = (rest / PER_VOWEL, rest % PER_VOWEL);
            let after = next.filter(|next| trailing(*next));
            // A syllable with no trailing consonant and one after it are
            // one syllable, when the face has that one.
            if trail == 0
                && let Some(after) = after.filter(|after| modern_trail(*after))
                && has_code(first + (after - TRAILS))
                && let Some(character) = char::from_u32(first + (after - TRAILS))
            {
                out.push(describe(character, cluster));
                at += 2;
                syllable_start = Some(before);
                continue;
            }
            // A syllable the face lacks, or one an old trailing consonant
            // follows, is drawn from its letters when the face has them.
            let known = has_code(first);
            let parts = [LEADS + lead, VOWELS + middle, TRAILS + trail];
            let count = if trail == 0 { 2 } else { 3 };
            if (!known || (trail == 0 && after.is_some()))
                && parts[..count].iter().all(|part| has_code(*part))
            {
                for (part, mask) in parts[..count].iter().zip([LEADING, VOWEL, TRAILING]) {
                    if let Some(character) = char::from_u32(*part) {
                        let mut letter = describe(character, cluster);
                        letter.mask |= mask;
                        letter.continues = mask != LEADING;
                        out.push(letter);
                    }
                }
                at += 1;
                // The old consonant after it is the syllable's own.
                if known && trail == 0 && after.is_some() {
                    let mut letter = glyphs[at].clone();
                    letter.mask |= TRAILING;
                    letter.cluster = cluster;
                    letter.continues = true;
                    out.push(letter);
                    at += 1;
                }
                syllable_start = Some(before);
                continue;
            }
        }
        // A syllable the face has, with nothing to add to it, is still a
        // syllable for a mark of pitch to go before.
        syllable_start = syllable(first).then_some(before);
        out.push(glyphs[at].clone());
        at += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{LEADING, TRAILING, VOWEL, compose};
    use crate::shaper::buffer::Glyph;

    /// A face of the syllables GA, GAG and HAN and the letters of Hangul.
    const FACE: &[u8] = include_bytes!("../../../font/fixtures/hangul.ttf");
    /// A face of the letters alone.
    const LETTERS: &[u8] = include_bytes!("../../../font/fixtures/hangul-letters.ttf");

    fn composed(face: &[u8], text: &str) -> Vec<(u32, usize, u32)> {
        let font = cutlass_font::read(face).expect("face");
        let glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| Glyph::new(character, at))
            .collect();
        compose(&font, glyphs, Glyph::new)
            .iter()
            .map(|glyph| {
                (
                    u32::from(glyph.character),
                    glyph.cluster,
                    glyph.mask & (LEADING | VOWEL | TRAILING),
                )
            })
            .collect()
    }

    #[test]
    fn letters_become_the_syllable_the_face_has() {
        // G A: GA. G A G: GAG. H A N: HAN.
        assert_eq!(composed(FACE, "\u{1100}\u{1161}"), [(0xAC00, 0, 0)]);
        assert_eq!(composed(FACE, "\u{1100}\u{1161}\u{11A8}"), [(0xAC01, 0, 0)]);
        assert_eq!(
            composed(FACE, "\u{1112}\u{1161}\u{11AB}\u{1100}\u{1161}"),
            [(0xD55C, 0, 0), (0xAC00, 9, 0)]
        );
        // A syllable and the trailing consonant after it.
        assert_eq!(composed(FACE, "\u{AC00}\u{11A8}"), [(0xAC01, 0, 0)]);
        // A syllable already whole, and other text, are left alone.
        assert_eq!(
            composed(FACE, "\u{AC01}a\u{1100}"),
            [(0xAC01, 0, 0), (0x61, 3, 0), (0x1100, 4, 0)]
        );
        // A syllable the face lacks stays as its letters, marked for the
        // face's own joins: N A.
        assert_eq!(
            composed(FACE, "\u{1102}\u{1161}"),
            [(0x1102, 0, LEADING), (0x1161, 0, VOWEL)]
        );
    }

    #[test]
    fn a_syllable_the_face_lacks_is_drawn_from_its_letters() {
        // The face of letters has no syllables: GAG is G, A, G.
        assert_eq!(
            composed(LETTERS, "\u{AC01}"),
            [
                (0x1100, 0, LEADING),
                (0x1161, 0, VOWEL),
                (0x11A8, 0, TRAILING)
            ]
        );
        assert_eq!(
            composed(LETTERS, "\u{AC00}"),
            [(0x1100, 0, LEADING), (0x1161, 0, VOWEL)]
        );
        // Letters stay letters, and take their marks.
        assert_eq!(
            composed(LETTERS, "\u{1100}\u{1161}\u{11A8}"),
            [
                (0x1100, 0, LEADING),
                (0x1161, 0, VOWEL),
                (0x11A8, 0, TRAILING)
            ]
        );
        // An old trailing consonant after a whole syllable takes the
        // syllable apart to join it: GA and an old final.
        assert_eq!(
            composed(FACE, "\u{AC00}\u{11C3}"),
            [
                (0x1100, 0, LEADING),
                (0x1161, 0, VOWEL),
                (0x11C3, 0, TRAILING)
            ]
        );
    }

    #[test]
    fn a_mark_of_pitch_goes_before_the_syllable_it_follows() {
        // The two dots have room in the face: before GA, in its cluster.
        assert_eq!(
            composed(FACE, "\u{AC00}\u{302F}"),
            [(0x302F, 0, 0), (0xAC00, 0, 0)]
        );
        // Before the whole of a syllable written in letters.
        assert_eq!(
            composed(FACE, "a\u{1102}\u{1161}\u{302F}"),
            [
                (0x61, 0, 0),
                (0x302F, 1, 0),
                (0x1102, 1, LEADING),
                (0x1161, 1, VOWEL)
            ]
        );
        // The one dot takes no room: it stays after.
        assert_eq!(
            composed(FACE, "\u{AC00}\u{302E}"),
            [(0xAC00, 0, 0), (0x302E, 3, 0)]
        );
        // Only the first mark after a syllable is the syllable's; the
        // next, and one after a letter that is no syllable, take a
        // dotted circle.
        assert_eq!(
            composed(FACE, "\u{AC00}\u{302F}\u{302F}"),
            [
                (0x302F, 0, 0),
                (0xAC00, 0, 0),
                (0x302F, 6, 0),
                (0x25CC, 6, 0)
            ]
        );
        assert_eq!(
            composed(FACE, "\u{1100}\u{302E}"),
            [(0x1100, 0, 0), (0x25CC, 3, 0), (0x302E, 3, 0)]
        );
        // A face with no circle leaves the mark as it is, and one with
        // no mark moves it all the same: its missing glyph has room.
        assert_eq!(composed(LETTERS, "\u{302F}"), [(0x302F, 0, 0)]);
        assert_eq!(
            composed(LETTERS, "\u{1100}\u{1161}\u{302E}"),
            [(0x302E, 0, 0), (0x1100, 0, LEADING), (0x1161, 0, VOWEL)]
        );
    }

    #[test]
    fn stray_letters_and_long_text_are_walked_once() {
        // A vowel with no consonant, a consonant with no vowel, and two
        // consonants: nothing to join.
        assert_eq!(
            composed(FACE, "\u{1161}\u{1100}\u{1100}"),
            [(0x1161, 0, 0), (0x1100, 3, 0), (0x1100, 6, 0)]
        );
        assert!(composed(FACE, "").is_empty());
        let started = std::time::Instant::now();
        let long = "\u{1112}\u{1161}\u{11AB}".repeat(50_000);
        assert_eq!(composed(FACE, &long).len(), 50_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}

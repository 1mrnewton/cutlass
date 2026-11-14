//! The scripts shaped by one shared set of rules: Tibetan, Sinhala,
//! Javanese, Balinese, Tai Tham, Cham, Sundanese and some sixty more.
//!
//! These scripts are written by syllable as Devanagari is, and what
//! differs between them is in Unicode's data and in their fonts, not in
//! the steps. So one set of rules serves them all, the Universal Shaping
//! Engine of the OpenType specification: each character has a category,
//! got from its Unicode properties; a run is cut into clusters by one
//! pattern over those categories; the font's forms are applied to each
//! cluster in a fixed order; a hook the font makes of a first `r` is
//! moved to the end of its cluster's base and what the font draws before
//! the base is moved to the front. The categories, the pattern and the
//! steps are the specification's, and give what HarfBuzz gives.

use cutlass_font::{Font, Tag};

use super::apply::Engine;
use super::buffer::{Buffer, Glyph};
use super::pattern::{Pattern, cut, each_syllable, either, maybe, merge, one, repeat, seq};
use super::syllables::insert_dotted_circles;
use super::{Feature, lookups};
use crate::TextError;
use crate::unicode::{gc, joining, props};

/// Categories of character. The numbers are the specification's.
pub(super) mod category {
    /// Anything else.
    pub(crate) const O: u8 = 0;
    /// A base letter.
    pub(crate) const B: u8 = 1;
    /// A number that joins others.
    pub(crate) const N: u8 = 4;
    /// A placeholder a mark may be shown on.
    pub(crate) const GB: u8 = 5;
    /// A joiner, a selector, or another mark that draws nothing.
    pub(crate) const CGJ: u8 = 6;
    /// A consonant written under another.
    pub(crate) const SUB: u8 = 11;
    /// A virama.
    pub(crate) const H: u8 = 12;
    /// A joiner of numbers.
    pub(crate) const HN: u8 = 13;
    pub(crate) const ZWNJ: u8 = 14;
    /// A word joiner, or a character with no meaning yet.
    pub(crate) const WJ: u8 = 16;
    /// A hook, or a letter written before its cluster.
    pub(crate) const R: u8 = 18;
    /// A vowel before the base, and a mark of a vowel there.
    pub(crate) const V_PRE: u8 = 22;
    pub(crate) const VM_PRE: u8 = 23;
    /// Final consonants above, below and after.
    pub(crate) const F_ABV: u8 = 24;
    pub(crate) const F_BLW: u8 = 25;
    pub(crate) const F_PST: u8 = 26;
    /// Medial consonants.
    pub(crate) const M_ABV: u8 = 27;
    pub(crate) const M_BLW: u8 = 28;
    pub(crate) const M_PST: u8 = 29;
    pub(crate) const M_PRE: u8 = 30;
    /// Marks that change a consonant, such as a nukta.
    pub(crate) const CM_ABV: u8 = 31;
    pub(crate) const CM_BLW: u8 = 32;
    /// Vowels.
    pub(crate) const V_ABV: u8 = 33;
    pub(crate) const V_BLW: u8 = 34;
    pub(crate) const V_PST: u8 = 35;
    /// Marks of a vowel: tones, nasal signs.
    pub(crate) const VM_ABV: u8 = 37;
    pub(crate) const VM_BLW: u8 = 38;
    pub(crate) const VM_PST: u8 = 39;
    /// Marks of a symbol.
    pub(crate) const SM_ABV: u8 = 41;
    pub(crate) const SM_BLW: u8 = 42;
    /// A consonant with the sign that stacks it.
    pub(crate) const CS: u8 = 43;
    /// A stacker that draws nothing.
    pub(crate) const IS: u8 = 44;
    /// Marks of a final consonant.
    pub(crate) const FM_ABV: u8 = 45;
    pub(crate) const FM_BLW: u8 = 46;
    pub(crate) const FM_PST: u8 = 47;
    /// The Tai Tham sakot.
    pub(crate) const SK: u8 = 48;
    /// Hieroglyphs and the controls that arrange them.
    pub(crate) const G: u8 = 49;
    pub(crate) const J: u8 = 50;
    pub(crate) const SB: u8 = 51;
    pub(crate) const SE: u8 = 52;
    /// A virama that is also a mark of a vowel.
    pub(crate) const HVM: u8 = 53;
    pub(crate) const HM: u8 = 54;
    pub(crate) const HR: u8 = 55;
    /// A virama that stops reordering.
    pub(crate) const RK: u8 = 56;
}

/// Kinds of cluster, kept in the low bits of a glyph's syllable number.
mod cluster {
    pub(super) const VIRAMA_ENDED: u8 = 1;
    pub(super) const SAKOT_ENDED: u8 = 2;
    pub(super) const STANDARD: u8 = 3;
    pub(super) const NUMBER_JOINER_ENDED: u8 = 4;
    pub(super) const NUMERAL: u8 = 5;
    pub(super) const SYMBOL: u8 = 6;
    pub(super) const HIEROGLYPH: u8 = 7;
    pub(super) const BROKEN: u8 = 8;
    pub(super) const NONE: u8 = 9;
}

// The feature each mask bit stands for.
const RPHF: u32 = 1 << 8;
const ISOL: u32 = 1 << 9;
const INIT: u32 = 1 << 10;
const MEDI: u32 = 1 << 11;
const FINA: u32 = 1 << 12;
const FORMS: u32 = ISOL | INIT | MEDI | FINA;

/// Characters whose category is not what their Unicode properties give:
/// the letters of scripts those properties do not cover, and the cases
/// the specification lists one by one. Each is a first and a last code
/// point and the category, in rising order.
const OVERRIDES: &[(u32, u32, u8)] = {
    use category::{
        B, CM_ABV, CM_BLW, G, GB, H, HM, HR, J, M_ABV, M_PST, O, SB, SE, SM_ABV, SUB, V_ABV, V_BLW,
        V_PST, VM_ABV, VM_BLW,
    };
    &[
        (0x005B, 0x005B, SB),
        (0x005D, 0x005D, SE),
        (0x007B, 0x007B, SB),
        (0x007D, 0x007D, SE),
        (0x07EB, 0x07F3, VM_ABV),
        (0x07FD, 0x07FD, VM_ABV),
        (0x0859, 0x085B, CM_BLW),
        (0x0F00, 0x0F01, B),
        (0x0F04, 0x0F06, B),
        (0x0F71, 0x0F71, CM_BLW),
        (0x0F72, 0x0F72, V_BLW),
        (0x0F74, 0x0F74, V_ABV),
        (0x0F7A, 0x0F7D, V_BLW),
        (0x0F7F, 0x0F7F, O),
        (0x0F80, 0x0F80, V_BLW),
        (0x0F86, 0x0F87, VM_ABV),
        (0x17CF, 0x17CF, VM_ABV),
        (0x1800, 0x1800, B),
        (0x1880, 0x1884, GB),
        (0x1885, 0x1886, CM_ABV),
        (0x18A9, 0x18A9, CM_BLW),
        (0x193A, 0x193A, VM_ABV),
        (0x1A18, 0x1A18, V_ABV),
        (0x1A7B, 0x1A7C, VM_ABV),
        (0x1A7F, 0x1A7F, VM_BLW),
        (0x1B6B, 0x1B73, SM_ABV),
        (0x1CF8, 0x1CF9, VM_ABV),
        (0x25CC, 0x25CC, B),
        (0x27E6, 0x27E6, SB),
        (0x27E7, 0x27E7, SE),
        (0x27E8, 0x27E8, SB),
        (0x27E9, 0x27E9, SE),
        (0x2D30, 0x2D67, B),
        (0x2D6F, 0x2D6F, B),
        (0x2D7F, 0x2D7F, H),
        (0x2E22, 0x2E22, SB),
        (0x2E23, 0x2E23, SE),
        (0x2E24, 0x2E24, SB),
        (0x2E25, 0x2E25, SE),
        (0xA9BE, 0xA9BE, M_PST),
        (0xAA29, 0xAA29, VM_ABV),
        (0xAA35, 0xAA35, M_ABV),
        (0xAAB0, 0xAAB0, V_BLW),
        (0xAAB2, 0xAAB3, V_BLW),
        (0xAAB4, 0xAAB4, V_ABV),
        (0xAAB7, 0xAAB8, V_BLW),
        (0xAABE, 0xAABE, V_BLW),
        (0x10570, 0x1057A, B),
        (0x1057C, 0x1058A, B),
        (0x1058C, 0x10592, B),
        (0x10594, 0x10595, B),
        (0x10597, 0x105A1, B),
        (0x105A3, 0x105B1, B),
        (0x105B3, 0x105B9, B),
        (0x105BB, 0x105BC, B),
        (0x10A0C, 0x10A0C, V_PST),
        (0x10A0D, 0x10A0D, VM_BLW),
        (0x10A38, 0x10A38, CM_BLW),
        (0x10AC6, 0x10AC6, B),
        (0x10ACB, 0x10ACC, B),
        (0x10AE2, 0x10AE3, B),
        (0x10AE5, 0x10AE6, CM_BLW),
        (0x10D24, 0x10D26, VM_ABV),
        (0x10D27, 0x10D27, CM_ABV),
        (0x10D30, 0x10D39, B),
        (0x10D4A, 0x10D65, B),
        (0x10D69, 0x10D6D, V_ABV),
        (0x10D6F, 0x10D85, B),
        (0x10E80, 0x10EA9, B),
        (0x10EAB, 0x10EAC, V_ABV),
        (0x10EB0, 0x10EB1, B),
        (0x10F45, 0x10F45, B),
        (0x10F46, 0x10F50, VM_BLW),
        (0x10F82, 0x10F85, CM_BLW),
        (0x11127, 0x11129, V_BLW),
        (0x1112A, 0x1112B, V_ABV),
        (0x1112D, 0x1112D, V_BLW),
        (0x11130, 0x11130, V_BLW),
        (0x11131, 0x11132, V_ABV),
        (0x11134, 0x11134, CM_ABV),
        (0x111DA, 0x111DA, B),
        (0x113CE, 0x113CE, VM_ABV),
        (0x113CF, 0x113CF, CM_BLW),
        (0x11942, 0x11942, M_PST),
        (0x11A3B, 0x11A3E, SUB),
        (0x13000, 0x1342F, G),
        (0x13430, 0x13436, J),
        (0x13437, 0x13437, SB),
        (0x13438, 0x13438, SE),
        (0x13439, 0x1343B, J),
        (0x1343C, 0x1343F, G),
        (0x13440, 0x13440, HR),
        (0x13441, 0x13446, G),
        (0x13447, 0x13455, HM),
        (0x13460, 0x143FA, G),
        (0x16AC0, 0x16AC9, B),
        (0x16B00, 0x16B2F, B),
        (0x16B30, 0x16B36, VM_ABV),
        (0x16F00, 0x16F4A, B),
        (0x16F4F, 0x16F4F, CM_BLW),
        (0x16F51, 0x16F87, V_BLW),
        (0x16F8F, 0x16F92, VM_BLW),
        (0x16FE4, 0x16FE4, B),
        (0x18B00, 0x18CD5, B),
        (0x18CFF, 0x18CFF, B),
        (0x1BC00, 0x1BC6A, B),
        (0x1BC70, 0x1BC7C, B),
        (0x1BC80, 0x1BC88, B),
        (0x1BC90, 0x1BC99, B),
        (0x1BC9D, 0x1BC9E, CM_BLW),
        (0x1E100, 0x1E12C, B),
        (0x1E130, 0x1E136, VM_ABV),
        (0x1E137, 0x1E13D, B),
        (0x1E140, 0x1E149, B),
        (0x1E14E, 0x1E14F, B),
        (0x1E290, 0x1E2AD, B),
        (0x1E2AE, 0x1E2AE, VM_ABV),
        (0x1E2C0, 0x1E2EB, B),
        (0x1E2EC, 0x1E2EF, VM_ABV),
        (0x1E2F0, 0x1E2F9, B),
        (0x1E4D0, 0x1E4EB, B),
        (0x1E4EC, 0x1E4EF, V_ABV),
        (0x1E4F0, 0x1E4F9, B),
        (0x1E5D0, 0x1E5ED, B),
        (0x1E5EE, 0x1E5EF, V_BLW),
        (0x1E5F0, 0x1E5FA, B),
        (0x1E944, 0x1E94A, CM_ABV),
        (0x1E94B, 0x1E94B, B),
        (0x1E950, 0x1E959, B),
    ]
};

/// The numbers Unicode's syllabic categories go by.
mod syllabic {
    pub(super) const OTHER: u8 = 0;
    pub(super) const AVAGRAHA: u8 = 1;
    pub(super) const BINDU: u8 = 2;
    pub(super) const BRAHMI_JOINING_NUMBER: u8 = 3;
    pub(super) const CANTILLATION_MARK: u8 = 4;
    pub(super) const CONSONANT: u8 = 5;
    pub(super) const CONSONANT_FINAL: u8 = 7;
    pub(super) const CONSONANT_HEAD_LETTER: u8 = 8;
    pub(super) const CONSONANT_INITIAL_POSTFIXED: u8 = 9;
    pub(super) const CONSONANT_KILLER: u8 = 10;
    pub(super) const CONSONANT_MEDIAL: u8 = 11;
    pub(super) const CONSONANT_PLACEHOLDER: u8 = 12;
    pub(super) const CONSONANT_PRECEDING_REPHA: u8 = 13;
    pub(super) const CONSONANT_PREFIXED: u8 = 14;
    pub(super) const CONSONANT_SUBJOINED: u8 = 15;
    pub(super) const CONSONANT_SUCCEEDING_REPHA: u8 = 16;
    pub(super) const CONSONANT_WITH_STACKER: u8 = 17;
    pub(super) const GEMINATION_MARK: u8 = 18;
    pub(super) const INVISIBLE_STACKER: u8 = 19;
    pub(super) const JOINER: u8 = 20;
    pub(super) const NON_JOINER: u8 = 22;
    pub(super) const NUKTA: u8 = 23;
    pub(super) const NUMBER: u8 = 24;
    pub(super) const NUMBER_JOINER: u8 = 25;
    pub(super) const PURE_KILLER: u8 = 26;
    pub(super) const REGISTER_SHIFTER: u8 = 27;
    pub(super) const SYLLABLE_MODIFIER: u8 = 28;
    pub(super) const TONE_LETTER: u8 = 29;
    pub(super) const TONE_MARK: u8 = 30;
    pub(super) const VIRAMA: u8 = 31;
    pub(super) const VISARGA: u8 = 32;
    pub(super) const VOWEL: u8 = 33;
    pub(super) const VOWEL_DEPENDENT: u8 = 34;
    pub(super) const VOWEL_INDEPENDENT: u8 = 35;
    pub(super) const REORDERING_KILLER: u8 = 36;
}

/// The numbers Unicode's positional categories go by.
mod positional {
    pub(super) const NONE: u8 = 0;
    pub(super) const BOTTOM: u8 = 1;
    pub(super) const BOTTOM_AND_LEFT: u8 = 2;
    pub(super) const BOTTOM_AND_RIGHT: u8 = 3;
    pub(super) const LEFT: u8 = 4;
    pub(super) const LEFT_AND_RIGHT: u8 = 5;
    pub(super) const OVERSTRUCK: u8 = 6;
    pub(super) const RIGHT: u8 = 7;
    pub(super) const TOP: u8 = 8;
    pub(super) const TOP_AND_BOTTOM: u8 = 9;
    pub(super) const TOP_AND_BOTTOM_AND_RIGHT: u8 = 10;
    pub(super) const TOP_AND_LEFT: u8 = 11;
    pub(super) const TOP_AND_LEFT_AND_RIGHT: u8 = 12;
    pub(super) const TOP_AND_RIGHT: u8 = 13;
    pub(super) const TOP_AND_BOTTOM_AND_LEFT: u8 = 15;
}

/// The category of `character`.
pub(super) fn categorize(character: char) -> u8 {
    let code = u32::from(character);
    let listed = OVERRIDES
        .binary_search_by(|(first, last, _)| {
            if code < *first {
                std::cmp::Ordering::Greater
            } else if code > *last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()
        .and_then(|at| OVERRIDES.get(at));
    match listed {
        Some((_, _, category)) => *category,
        None => derived(character),
    }
}

/// The category a character's Unicode properties give it: first what it
/// is, a base or a vowel or a mark of one, then for the kinds that sit
/// around the base, which side it is on.
fn derived(character: char) -> u8 {
    use category::{
        B, CGJ, CM_ABV, CM_BLW, CS, F_ABV, F_BLW, F_PST, FM_ABV, FM_BLW, FM_PST, GB, H, HN, HVM,
        IS, M_ABV, M_BLW, M_PRE, M_PST, N, O, R, RK, SK, SUB, V_ABV, V_BLW, V_PRE, V_PST, VM_ABV,
        VM_BLW, VM_PRE, VM_PST, WJ, ZWNJ,
    };
    use positional as side;
    use syllabic as kind;
    let code = u32::from(character);
    let props = props(character);
    let category = props.category();
    let joins = matches!(
        props.joining(),
        joining::CAUSING | joining::DUAL | joining::LEFT | joining::RIGHT
    );
    let ignorable = props.default_ignorable();
    let mut said = props.syllabic();
    let mut place = props.positional();
    // A character Unicode says nothing of here is anything else.
    if said == kind::OTHER && place == side::NONE && !joins && !ignorable {
        return O;
    }
    // These scripts have shaping of their own.
    if matches!(
        &props.script(),
        b"Arab" | b"Laoo" | b"Samr" | b"Syrc" | b"Thai"
    ) {
        return O;
    }
    // Marks Unicode gives a place and no kind.
    match code {
        0x1CE2..=0x1CE8 => said = kind::CANTILLATION_MARK,
        0x0F18 | 0x0F19 | 0x0F3E | 0x0F3F => said = kind::VOWEL_DEPENDENT,
        0x1CED => said = kind::TONE_MARK,
        0x11302 | 0x11303 | 0x114C1 => place = side::TOP,
        _ => {}
    }
    let letter = category == gc::OTHER_LETTER;
    let mark = matches!(
        category,
        gc::NONSPACING_MARK | gc::ENCLOSING_MARK | gc::SPACING_MARK
    );
    let joiner = said == kind::JOINER || (ignorable && mark);
    let base = matches!(
        said,
        kind::NUMBER
            | kind::CONSONANT
            | kind::CONSONANT_HEAD_LETTER
            | kind::TONE_LETTER
            | kind::VOWEL_INDEPENDENT
    ) || (joins && said != kind::JOINER)
        || (letter
            && matches!(
                said,
                kind::AVAGRAHA
                    | kind::BINDU
                    | kind::CONSONANT_FINAL
                    | kind::CONSONANT_MEDIAL
                    | kind::CONSONANT_SUBJOINED
                    | kind::VOWEL
                    | kind::VOWEL_DEPENDENT
            ));
    let placeholder =
        said == kind::CONSONANT_PLACEHOLDER || matches!(code, 0x2015 | 0x2022 | 0x25FB..=0x25FE);
    // The fillers of Hangul and of shorthand draw nothing and join nothing.
    let filler = matches!(code, 0x115F | 0x1160 | 0x3164 | 0xFFA0 | 0x1BCA0..=0x1BCA3);
    let word_joiner =
        (ignorable && !filler && said == kind::OTHER && !joiner) || category == gc::UNASSIGNED;
    if base {
        return B;
    }
    if placeholder {
        return GB;
    }
    if joiner {
        return CGJ;
    }
    if word_joiner {
        return WJ;
    }
    // Final consonants, medials, marks and vowels take a side.
    let sided = |sides: &[(&[u8], u8)]| {
        sides
            .iter()
            .find(|(places, _)| places.contains(&place))
            .map_or(O, |(_, category)| *category)
    };
    match said {
        kind::BRAHMI_JOINING_NUMBER => N,
        kind::CONSONANT_FINAL | kind::CONSONANT_SUCCEEDING_REPHA => sided(&[
            (&[side::TOP], F_ABV),
            (&[side::BOTTOM], F_BLW),
            (&[side::RIGHT], F_PST),
        ]),
        kind::SYLLABLE_MODIFIER => sided(&[
            (&[side::TOP], FM_ABV),
            (&[side::BOTTOM], FM_BLW),
            (&[side::NONE], FM_PST),
        ]),
        kind::CONSONANT_MEDIAL | kind::CONSONANT_INITIAL_POSTFIXED => sided(&[
            (&[side::TOP], M_ABV),
            (
                &[side::BOTTOM, side::BOTTOM_AND_LEFT, side::BOTTOM_AND_RIGHT],
                M_BLW,
            ),
            (&[side::RIGHT], M_PST),
            (&[side::LEFT, side::TOP_AND_BOTTOM_AND_LEFT], M_PRE),
        ]),
        kind::NUKTA | kind::GEMINATION_MARK | kind::CONSONANT_KILLER => sided(&[
            (&[side::TOP], CM_ABV),
            (&[side::BOTTOM, side::OVERSTRUCK], CM_BLW),
        ]),
        kind::CONSONANT_SUBJOINED => SUB,
        kind::CONSONANT_WITH_STACKER => CS,
        // The Sinhala virama is also a mark of a vowel.
        kind::VIRAMA if code == 0x0DCA => HVM,
        kind::VIRAMA => H,
        kind::NUMBER_JOINER => HN,
        kind::INVISIBLE_STACKER if code == 0x1A60 => SK,
        kind::INVISIBLE_STACKER => IS,
        kind::NON_JOINER => ZWNJ,
        kind::REORDERING_KILLER => RK,
        kind::CONSONANT_PRECEDING_REPHA | kind::CONSONANT_PREFIXED => R,
        kind::PURE_KILLER | kind::VOWEL | kind::VOWEL_DEPENDENT => sided(&[
            (
                &[
                    side::TOP,
                    side::TOP_AND_BOTTOM,
                    side::TOP_AND_BOTTOM_AND_RIGHT,
                    side::TOP_AND_RIGHT,
                ],
                V_ABV,
            ),
            (
                &[side::BOTTOM, side::OVERSTRUCK, side::BOTTOM_AND_RIGHT],
                V_BLW,
            ),
            (&[side::RIGHT], V_PST),
            (
                &[
                    side::LEFT,
                    side::TOP_AND_LEFT,
                    side::TOP_AND_LEFT_AND_RIGHT,
                    side::LEFT_AND_RIGHT,
                ],
                V_PRE,
            ),
        ]),
        kind::TONE_MARK
        | kind::CANTILLATION_MARK
        | kind::REGISTER_SHIFTER
        | kind::VISARGA
        | kind::BINDU => sided(&[
            (&[side::TOP], VM_ABV),
            (&[side::BOTTOM, side::OVERSTRUCK], VM_BLW),
            (&[side::RIGHT], VM_PST),
            (&[side::LEFT], VM_PRE),
        ]),
        _ => O,
    }
}

/// Set each glyph's category.
pub(crate) fn describe_all(glyphs: &mut [Glyph]) {
    for glyph in glyphs {
        glyph.kind = categorize(glyph.character);
    }
}

/// The kinds of cluster, each with its pattern, in the order of choice
/// between two that match the same text.
fn grammar() -> Vec<(u8, Pattern)> {
    use category::{
        B, CM_ABV, CM_BLW, CS, F_ABV, F_BLW, F_PST, FM_ABV, FM_BLW, FM_PST, G, GB, H, HM, HN, HR,
        HVM, IS, J, M_ABV, M_BLW, M_PRE, M_PST, N, O, R, RK, SB, SE, SK, SM_ABV, SM_BLW, SUB,
        V_ABV, V_BLW, V_PRE, V_PST, VM_ABV, VM_BLW, VM_PRE, VM_PST, ZWNJ,
    };
    let many = |kind: u8| repeat(one(&[kind]));
    let some = |kind: u8| seq(vec![one(&[kind]), repeat(one(&[kind]))]);
    let halant = || one(&[H, HVM, IS, SK]);
    let consonant_modifiers = || {
        seq(vec![
            many(CM_ABV),
            many(CM_BLW),
            repeat(seq(vec![
                either(vec![seq(vec![halant(), one(&[B])]), one(&[SUB])]),
                many(CM_ABV),
                many(CM_BLW),
            ])),
        ])
    };
    let medials = || {
        seq(vec![
            maybe(one(&[M_PRE])),
            maybe(one(&[M_ABV])),
            maybe(one(&[M_BLW])),
            maybe(one(&[M_PST])),
        ])
    };
    let vowels = || {
        either(vec![
            seq(vec![many(V_PRE), many(V_ABV), many(V_BLW), many(V_PST)]),
            one(&[H]),
        ])
    };
    let vowel_modifiers = || {
        seq(vec![
            maybe(one(&[HVM])),
            many(VM_PRE),
            many(VM_ABV),
            many(VM_BLW),
            many(VM_PST),
        ])
    };
    let finals = || seq(vec![many(F_ABV), many(F_BLW), many(F_PST)]);
    let final_modifiers = || {
        either(vec![
            seq(vec![many(FM_ABV), many(FM_BLW)]),
            maybe(one(&[FM_PST])),
        ])
    };
    let start = || seq(vec![maybe(one(&[R, CS])), one(&[B, GB])]);
    let middle = || {
        seq(vec![
            consonant_modifiers(),
            medials(),
            vowels(),
            vowel_modifiers(),
            repeat(seq(vec![one(&[SK]), one(&[B])])),
        ])
    };
    let complex_tail = || seq(vec![middle(), finals(), final_modifiers()]);
    let number_joiner_tail = || seq(vec![repeat(seq(vec![one(&[HN]), one(&[N])])), one(&[HN])]);
    let numeral_tail = || {
        seq(vec![
            one(&[HN]),
            one(&[N]),
            repeat(seq(vec![one(&[HN]), one(&[N])])),
        ])
    };
    let symbol_tail = || either(vec![seq(vec![some(SM_ABV), many(SM_BLW)]), some(SM_BLW)]);
    let virama_tail = || seq(vec![consonant_modifiers(), one(&[IS, RK])]);
    let sakot_tail = || seq(vec![middle(), one(&[SK])]);
    let tail = || {
        either(vec![
            complex_tail(),
            sakot_tail(),
            symbol_tail(),
            virama_tail(),
        ])
    };
    // A non-joiner after a cluster is part of it.
    let ended = |pattern: Pattern| seq(vec![pattern, maybe(one(&[ZWNJ]))]);
    let glyph = || {
        seq(vec![
            one(&[G]),
            maybe(one(&[HR])),
            maybe(one(&[HM])),
            many(SE),
        ])
    };
    vec![
        (
            cluster::VIRAMA_ENDED,
            ended(seq(vec![start(), virama_tail()])),
        ),
        (
            cluster::SAKOT_ENDED,
            ended(seq(vec![start(), sakot_tail()])),
        ),
        (cluster::STANDARD, ended(seq(vec![start(), complex_tail()]))),
        (
            cluster::NUMBER_JOINER_ENDED,
            ended(seq(vec![one(&[N]), number_joiner_tail()])),
        ),
        (
            cluster::NUMERAL,
            ended(seq(vec![one(&[N]), maybe(numeral_tail())])),
        ),
        (
            cluster::SYMBOL,
            ended(seq(vec![one(&[O, GB, SB]), maybe(tail())])),
        ),
        (
            cluster::HIEROGLYPH,
            ended(seq(vec![
                many(SB),
                glyph(),
                repeat(seq(vec![one(&[J]), many(SB), maybe(glyph())])),
            ])),
        ),
        (cluster::NONE, one(&[FM_PST])),
        (
            cluster::BROKEN,
            ended(seq(vec![
                maybe(one(&[R])),
                either(vec![tail(), number_joiner_tail(), numeral_tail()]),
            ])),
        ),
    ]
}

/// Number each glyph's cluster. `true` when a cluster has no base to
/// stand on.
///
/// Joiners, selectors and the like are not part of the pattern: each
/// stays in the cluster of the glyph before it. Nor is a non-joiner
/// before a mark, which only keeps the mark from being reshaped.
fn find_clusters(glyphs: &mut [Glyph]) -> bool {
    let skipped = |glyph: &Glyph| glyph.kind == category::CGJ;
    let places: Vec<usize> = (0..glyphs.len())
        .filter(|at| {
            let glyph = &glyphs[*at];
            if skipped(glyph) {
                return false;
            }
            if glyph.kind != category::ZWNJ {
                return true;
            }
            glyphs[*at + 1..]
                .iter()
                .find(|after| !skipped(after))
                .is_none_or(|after| {
                    !matches!(
                        after.category,
                        gc::NONSPACING_MARK | gc::ENCLOSING_MARK | gc::SPACING_MARK
                    )
                })
        })
        .collect();
    let kinds: Vec<u8> = places.iter().map(|at| glyphs[*at].kind).collect();
    let cuts = cut(&kinds, &grammar(), cluster::NONE);
    let mut serial = 1_u8;
    let mut start = places.first().copied().unwrap_or(glyphs.len());
    for (end, kind) in &cuts {
        let end = places.get(*end).copied().unwrap_or(glyphs.len());
        for glyph in &mut glyphs[start..end] {
            glyph.syllable = (serial << 4) | kind;
        }
        serial = if serial == 15 { 1 } else { serial + 1 };
        start = end;
    }
    cuts.iter().any(|(_, kind)| *kind == cluster::BROKEN)
}

/// Say which glyphs a hook may be made of, and which form each cluster
/// takes from the clusters around it.
fn set_masks(glyphs: &mut [Glyph]) {
    // A hook is made of the first three glyphs of a cluster, or of the
    // one character that is a hook already.
    each_syllable(glyphs, |glyphs, start, end| {
        let count = if glyphs[start].kind == category::R {
            1
        } else {
            (end - start).min(3)
        };
        for glyph in &mut glyphs[start..start + count] {
            glyph.mask |= RPHF;
        }
    });
    // A cluster alone is isolated; the first of a row is initial, the
    // last final, and the ones between medial.
    let mut last: Option<(usize, usize, u32)> = None;
    let mut start = 0;
    while start < glyphs.len() {
        let syllable = glyphs[start].syllable;
        let mut end = start + 1;
        while end < glyphs.len() && glyphs[end].syllable == syllable {
            end += 1;
        }
        let kind = syllable & 0x0F;
        if kind == cluster::HIEROGLYPH || kind == cluster::NONE {
            last = None;
        } else {
            let form = match last {
                Some((from, to, form)) => {
                    let earlier = if form == FINA { MEDI } else { INIT };
                    for glyph in &mut glyphs[from..to] {
                        glyph.mask = (glyph.mask & !FORMS) | earlier;
                    }
                    FINA
                }
                None => ISOL,
            };
            for glyph in &mut glyphs[start..end] {
                glyph.mask = (glyph.mask & !FORMS) | form;
            }
            last = Some((start, end, form));
        }
        start = end;
    }
}

/// Run the substitutions for a run of one of these scripts. `ligatures`
/// makes the ones a reader can do without.
pub(crate) fn substitute<'a>(
    font: &Font<'a>,
    mut engine: Option<Engine<'_, 'a>>,
    buffer: &mut Buffer,
    scripts: &[Tag],
    ligatures: bool,
) -> Result<(), TextError> {
    let table = font.gsub();
    let broken = {
        let glyphs = buffer.glyphs_mut();
        let broken = find_clusters(glyphs);
        set_masks(glyphs);
        broken
    };
    let mut stage = |buffer: &mut Buffer, features: &[Feature]| -> Result<(), TextError> {
        let (Some(engine), Some(table)) = (engine.as_mut(), table) else {
            return Ok(());
        };
        for mut lookup in lookups(table, scripts, features)? {
            lookup.per_syllable = true;
            engine.apply(buffer, &lookup)?;
        }
        Ok(())
    };
    let forget = |buffer: &mut Buffer| {
        for glyph in buffer.glyphs_mut().iter_mut() {
            glyph.substituted = false;
        }
    };
    let every = |tag: &[u8; 4]| Feature::every(tag).manual();
    stage(
        buffer,
        &[
            Feature::every(b"locl"),
            Feature::every(b"ccmp"),
            Feature::every(b"nukt"),
            every(b"akhn"),
        ],
    )?;
    // A glyph the hook feature makes is a hook, wherever it is.
    forget(buffer);
    stage(buffer, &[Feature::masked(b"rphf", RPHF).manual()])?;
    each_syllable(buffer.glyphs_mut(), |glyphs, start, end| {
        for glyph in &mut glyphs[start..end] {
            if glyph.mask & RPHF == 0 {
                break;
            }
            if glyph.substituted {
                glyph.kind = category::R;
                break;
            }
        }
    });
    // And one the pre-base form makes goes where a vowel before the
    // base goes.
    forget(buffer);
    stage(buffer, &[every(b"pref")])?;
    each_syllable(buffer.glyphs_mut(), |glyphs, start, end| {
        if let Some(glyph) = glyphs[start..end]
            .iter_mut()
            .find(|glyph| glyph.substituted)
        {
            glyph.kind = category::V_PRE;
        }
    });
    stage(
        buffer,
        &[
            every(b"rkrf"),
            every(b"abvf"),
            every(b"blwf"),
            every(b"half"),
            every(b"pstf"),
            every(b"vatu"),
            every(b"cjct"),
        ],
    )?;
    {
        let glyphs = buffer.glyphs_mut();
        if broken {
            insert_dotted_circles(
                font,
                glyphs,
                cluster::BROKEN,
                category::B,
                Some(category::R),
            );
        }
        each_syllable(glyphs, reorder);
        for glyph in glyphs.iter_mut() {
            glyph.syllable = 0;
        }
    }
    let (Some(engine), Some(table)) = (engine.as_mut(), table) else {
        return Ok(());
    };
    // The forms a cluster takes from its neighbours.
    let forms = [
        Feature::masked(b"isol", ISOL),
        Feature::masked(b"init", INIT),
        Feature::masked(b"medi", MEDI),
        Feature::masked(b"fina", FINA),
    ];
    for lookup in lookups(table, scripts, &forms)? {
        engine.apply(buffer, &lookup)?;
    }
    // The forms that span a cluster, with what every script shares.
    let mut last = vec![
        every(b"abvs"),
        every(b"blws"),
        every(b"haln"),
        every(b"pres"),
        every(b"psts"),
        Feature::every(b"rlig"),
        Feature::every(b"calt"),
        Feature::every(b"rclt"),
    ];
    if ligatures {
        last.extend([Feature::every(b"clig"), Feature::every(b"liga")]);
    }
    for lookup in lookups(table, scripts, &last)? {
        engine.apply(buffer, &lookup)?;
    }
    Ok(())
}

/// A virama the font has not joined to anything.
fn is_halant(glyph: &Glyph) -> bool {
    !glyph.ligated && matches!(glyph.kind, category::H | category::HVM | category::IS)
}

/// What is written after the base: a hook is put before the first of
/// these.
fn after_base(glyph: &Glyph) -> bool {
    use category::{
        F_ABV, F_BLW, F_PST, FM_ABV, FM_BLW, FM_PST, M_ABV, M_BLW, M_PRE, M_PST, V_ABV, V_BLW,
        V_PRE, V_PST, VM_ABV, VM_BLW, VM_PRE, VM_PST,
    };
    matches!(
        glyph.kind,
        F_ABV
            | F_BLW
            | F_PST
            | FM_ABV
            | FM_BLW
            | FM_PST
            | M_ABV
            | M_BLW
            | M_PST
            | M_PRE
            | V_ABV
            | V_BLW
            | V_PST
            | V_PRE
            | VM_ABV
            | VM_BLW
            | VM_PST
            | VM_PRE
    )
}

/// Put a cluster's glyphs in the order they are written: a hook after
/// the base and what is stacked on it, and what is drawn before the base
/// at the front, or after the last virama before it.
fn reorder(glyphs: &mut [Glyph], start: usize, end: usize) {
    if !matches!(
        glyphs[start].syllable & 0x0F,
        cluster::VIRAMA_ENDED
            | cluster::SAKOT_ENDED
            | cluster::STANDARD
            | cluster::SYMBOL
            | cluster::BROKEN
    ) {
        return;
    }
    if glyphs[start].kind == category::R && end - start > 1 {
        for index in start + 1..end {
            let after = after_base(&glyphs[index]) || is_halant(&glyphs[index]);
            if after || index == end - 1 {
                // Before the glyph after the base, or at the very end.
                let to = if after { index - 1 } else { index };
                merge(glyphs, start, to + 1);
                glyphs[start..=to].rotate_left(1);
                break;
            }
        }
    }
    let mut front = start;
    for index in start..end {
        if is_halant(&glyphs[index]) {
            front = index + 1;
        } else if matches!(glyphs[index].kind, category::V_PRE | category::VM_PRE)
            // Of several glyphs made from one, only the first moves.
            && glyphs[index].lig_part == 0
            && front < index
        {
            merge(glyphs, front, index + 1);
            glyphs[front..=index].rotate_right(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{categorize, category, cluster, find_clusters, reorder};
    use crate::shaper::buffer::Glyph;
    use crate::shaper::pattern::each_syllable;

    fn glyphs_of(text: &str) -> Vec<Glyph> {
        text.char_indices()
            .map(|(at, character)| {
                let mut glyph = Glyph::new(character, at);
                glyph.kind = categorize(character);
                glyph.category = crate::unicode::props(character).category();
                glyph
            })
            .collect()
    }

    /// The clusters of `text`: each one's characters and its kind.
    fn clusters(text: &str) -> Vec<(String, u8)> {
        let mut glyphs = glyphs_of(text);
        let _broken = find_clusters(&mut glyphs);
        let mut found: Vec<(String, u8, u8)> = Vec::new();
        for glyph in &glyphs {
            match found.last_mut().filter(|last| last.2 == glyph.syllable) {
                Some(last) => last.0.push(glyph.character),
                None => found.push((
                    glyph.character.to_string(),
                    glyph.syllable & 0x0F,
                    glyph.syllable,
                )),
            }
        }
        found
            .into_iter()
            .map(|(text, kind, _)| (text, kind))
            .collect()
    }

    /// `text` after its clusters are put in written order.
    fn reordered(text: &str) -> String {
        let mut glyphs = glyphs_of(text);
        let _broken = find_clusters(&mut glyphs);
        each_syllable(&mut glyphs, reorder);
        glyphs.iter().map(|glyph| glyph.character).collect()
    }

    #[test]
    fn characters_have_the_category_their_properties_give() {
        // Tibetan: a letter, a subjoined letter, vowels above and below.
        assert_eq!(categorize('\u{F40}'), category::B);
        assert_eq!(categorize('\u{F90}'), category::SUB);
        assert_eq!(categorize('\u{F72}'), category::V_BLW);
        assert_eq!(categorize('\u{F74}'), category::V_ABV);
        // Sinhala: its virama is also a mark of a vowel.
        assert_eq!(categorize('\u{D9A}'), category::B);
        assert_eq!(categorize('\u{DCA}'), category::HVM);
        assert_eq!(categorize('\u{DD9}'), category::V_PRE);
        assert_eq!(categorize('\u{D82}'), category::VM_PST);
        // Javanese: a hook, a medial drawn before the base, a virama.
        assert_eq!(categorize('\u{A982}'), category::F_ABV);
        assert_eq!(categorize('\u{A9BF}'), category::M_BLW);
        assert_eq!(categorize('\u{A9BA}'), category::V_PRE);
        assert_eq!(categorize('\u{A9C0}'), category::H);
        // Tai Tham: the sakot, and a stacker that draws nothing elsewhere.
        assert_eq!(categorize('\u{1A60}'), category::SK);
        assert_eq!(categorize('\u{1039}'), category::IS);
        // Shared characters.
        assert_eq!(categorize('\u{200C}'), category::ZWNJ);
        assert_eq!(categorize('\u{200D}'), category::CGJ);
        assert_eq!(categorize('\u{FE00}'), category::CGJ);
        assert_eq!(categorize('\u{25CC}'), category::B);
        assert_eq!(categorize('\u{A0}'), category::GB);
        assert_eq!(categorize('\u{2060}'), category::WJ);
        assert_eq!(categorize('a'), category::O);
        assert_eq!(categorize(' '), category::O);
        // Scripts with shaping of their own are left to it.
        assert_eq!(categorize('\u{E01}'), category::O);
        assert_eq!(categorize('\u{628}'), category::O);
        assert_eq!(categorize(char::MAX), category::O);
    }

    #[test]
    fn every_character_has_the_category_harfbuzz_gives_it() {
        // One number for the categories of every code point up to the
        // end of the tag characters, as HarfBuzz 14's table for Unicode
        // 17 has them: a change to the rules or to the list of
        // exceptions shows here.
        let mut sum = 0xcbf2_9ce4_8422_2325_u64;
        for code in 0..0xE_1000_u32 {
            // A surrogate is no character; the table has it as nothing.
            let category = char::from_u32(code).map_or(category::O, categorize);
            sum ^= u64::from(category);
            sum = sum.wrapping_mul(0x0100_0000_01b3);
        }
        assert_eq!(sum, 0xDB75_6DD8_6884_9F72);
    }

    #[test]
    fn text_is_cut_into_clusters() {
        // Javanese: ka with a vowel before it, then ka, virama, ka.
        assert_eq!(
            clusters("\u{A98F}\u{A9BA}\u{A98F}\u{A9C0}\u{A98F}"),
            [
                ("\u{A98F}\u{A9BA}".to_owned(), cluster::STANDARD),
                ("\u{A98F}\u{A9C0}\u{A98F}".to_owned(), cluster::STANDARD),
            ]
        );
        // A Tibetan stack with a vowel is one cluster.
        assert_eq!(clusters("\u{F66}\u{F92}\u{FB2}\u{F74}").len(), 1);
        // A vowel with no base is broken, and so is a lone non-joiner.
        assert_eq!(clusters("\u{A9BA}")[0].1, cluster::BROKEN);
        assert_eq!(clusters("\u{200C}")[0].1, cluster::BROKEN);
        // A non-joiner after a cluster ends it and stays in it; a joiner
        // is in the cluster of the glyph before it, whatever follows.
        assert_eq!(
            clusters("\u{A98F}\u{200C}\u{A98F}"),
            [
                ("\u{A98F}\u{200C}".to_owned(), cluster::STANDARD),
                ("\u{A98F}".to_owned(), cluster::STANDARD),
            ]
        );
        assert_eq!(clusters("\u{A98F}\u{200D}\u{A9BA}").len(), 1);
        // A Sinhala letter with its virama, which here ends the cluster
        // as a mark of the vowel.
        assert_eq!(clusters("\u{D9A}\u{DCA}")[0].1, cluster::STANDARD);
        // Latin and spaces are symbols on their own.
        assert_eq!(
            clusters("a "),
            [
                ("a".to_owned(), cluster::SYMBOL),
                (" ".to_owned(), cluster::SYMBOL)
            ]
        );
        assert!(clusters("").is_empty());
    }

    #[test]
    fn what_is_drawn_before_the_base_is_moved_there() {
        // A vowel before the base goes to the front of its cluster.
        assert_eq!(reordered("\u{A98F}\u{A9BA}"), "\u{A9BA}\u{A98F}");
        // But not past a virama: it goes before the last consonant.
        assert_eq!(
            reordered("\u{A98F}\u{A9C0}\u{A98F}\u{A9BA}"),
            "\u{A98F}\u{A9C0}\u{A9BA}\u{A98F}"
        );
        // Each cluster on its own, and other text left alone.
        assert_eq!(
            reordered("a\u{A98F}\u{A9BA}\u{A98F}\u{A9BA}"),
            "a\u{A9BA}\u{A98F}\u{A9BA}\u{A98F}"
        );
        // A vowel after a space or a symbol is drawn before that.
        assert_eq!(reordered("a \u{A9BA}"), "a\u{A9BA} ");
        // A hook written first goes after the base, before its vowels.
        let mut glyphs = glyphs_of("\u{A98F}\u{A98F}\u{A9B6}");
        glyphs[0].kind = category::R;
        let _broken = find_clusters(&mut glyphs);
        // The hook starts a cluster with the letter after it.
        assert_eq!(glyphs[0].syllable, glyphs[1].syllable);
        each_syllable(&mut glyphs, reorder);
        assert_eq!(glyphs[1].kind, category::R);
        assert_eq!(glyphs[2].character, '\u{A9B6}');
    }

    #[test]
    fn a_long_run_is_cut_and_ordered_in_time() {
        let started = std::time::Instant::now();
        for unit in [
            "\u{A98F}\u{A9C0}\u{A98F}\u{A9BA}",
            "\u{A9BA}",
            "\u{A9C0}",
            "\u{200C}",
        ] {
            let text = unit.repeat(10_000);
            assert!(!reordered(&text).is_empty());
        }
        // One base under thousands of marks is one cluster, and so is a
        // stack of thousands of consonants.
        let mut heavy = String::from("\u{A98F}");
        heavy.push_str(&"\u{A981}".repeat(20_000));
        assert_eq!(clusters(&heavy).len(), 1);
        let stack = "\u{A98F}\u{A9C0}".repeat(10_000);
        assert_eq!(clusters(&stack).len(), 1);
        assert_eq!(reordered(&stack).chars().count(), 20_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
    }
}

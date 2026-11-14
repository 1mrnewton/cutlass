//! Devanagari and the eight scripts shaped like it.
//!
//! These scripts are written by syllable. A syllable has one main
//! consonant; the consonants before it are drawn as half forms or joined
//! to it, and the ones after it go below or behind. A vowel sign may be
//! written before the consonants it is spoken after, and an `r` that
//! starts a syllable is drawn as a hook over its end. So the text is cut
//! into syllables, each one's parts are put in the order they are written,
//! the font's forms are applied a feature at a time, and what the forms
//! made is put in its last place. The steps follow the OpenType
//! specification for these scripts.

use cutlass_font::{Font, GlyphClass, LayoutTable, Substitution, Tag};

use super::apply::Engine;
use super::buffer::{Buffer, Glyph};
use super::pattern::{Pattern, each_syllable, maybe, merge, one, repeat, seq};
use super::{Feature, lookups};
use crate::TextError;
use crate::script::Script;
use crate::unicode::{gc, props};

/// Kinds of character, a bit each so a set of them is a mask.
mod kind {
    pub(super) const OTHER: u8 = 0;
    pub(super) const CONSONANT: u8 = 1;
    pub(super) const VOWEL: u8 = 2;
    pub(super) const NUKTA: u8 = 3;
    pub(super) const HALANT: u8 = 4;
    pub(super) const NON_JOINER: u8 = 5;
    pub(super) const JOINER: u8 = 6;
    pub(super) const MATRA: u8 = 7;
    pub(super) const MODIFIER: u8 = 8;
    pub(super) const ACCENT: u8 = 9;
    pub(super) const PLACEHOLDER: u8 = 10;
    pub(super) const DOTTED_CIRCLE: u8 = 11;
    pub(super) const REPHA: u8 = 14;
    pub(super) const RA: u8 = 15;
    pub(super) const MEDIAL: u8 = 16;
    pub(super) const SYMBOL: u8 = 17;
    pub(super) const STACKER: u8 = 18;
}

/// Where in its syllable a character goes; sorting by this is the first
/// reordering.
mod place {
    pub(super) const START: u8 = 0;
    pub(super) const RA_TO_REPH: u8 = 1;
    pub(super) const PRE_MATRA: u8 = 2;
    pub(super) const PRE_CONSONANT: u8 = 3;
    pub(super) const BASE: u8 = 4;
    pub(super) const AFTER_MAIN: u8 = 5;
    pub(super) const BEFORE_SUB: u8 = 7;
    pub(super) const BELOW: u8 = 8;
    pub(super) const AFTER_SUB: u8 = 9;
    pub(super) const POST: u8 = 11;
    pub(super) const AFTER_POST: u8 = 12;
    pub(super) const MODIFIER: u8 = 13;
    pub(super) const END: u8 = 14;
}

/// Kinds of syllable, kept in the low bits of a glyph's syllable number.
mod syllable {
    pub(super) const CONSONANT: u8 = 0;
    pub(super) const VOWEL: u8 = 1;
    pub(super) const STANDALONE: u8 = 2;
    pub(super) const SYMBOL: u8 = 3;
    pub(super) const BROKEN: u8 = 4;
    pub(super) const OTHER: u8 = 5;
}

const fn bit(kind: u8) -> u32 {
    1 << kind
}

const CONSONANTS: u32 = bit(kind::CONSONANT)
    | bit(kind::STACKER)
    | bit(kind::RA)
    | bit(kind::MEDIAL)
    | bit(kind::VOWEL)
    | bit(kind::PLACEHOLDER)
    | bit(kind::DOTTED_CIRCLE);
const JOINERS: u32 = bit(kind::JOINER) | bit(kind::NON_JOINER);

// The feature each mask bit stands for.
const RPHF: u32 = 1 << 8;
const PREF: u32 = 1 << 9;
const BLWF: u32 = 1 << 10;
const ABVF: u32 = 1 << 11;
const HALF: u32 = 1 << 12;
const PSTF: u32 = 1 << 13;
const INIT: u32 = 1 << 14;

/// Where a script puts the hook an initial `r` becomes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RephPlace {
    AfterMain,
    BeforeSub,
    AfterSub,
    BeforePost,
    AfterPost,
}

/// How a script writes that hook.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RephMode {
    /// `r` and the virama.
    Implicit,
    /// `r`, the virama and a joiner.
    Explicit,
    /// Its own character.
    Logical,
}

/// What differs from one of these scripts to the next.
#[derive(Clone, Copy)]
struct Config {
    virama: char,
    reph_place: RephPlace,
    reph_mode: RephMode,
    /// Below-base forms are made before the main consonant too.
    below_before: bool,
}

fn config(script: Script) -> Config {
    let (virama, reph_place, reph_mode, below_before) = match &script.0 {
        b"Beng" => ('\u{9CD}', RephPlace::AfterSub, RephMode::Implicit, true),
        b"Guru" => ('\u{A4D}', RephPlace::BeforeSub, RephMode::Implicit, true),
        b"Gujr" => ('\u{ACD}', RephPlace::BeforePost, RephMode::Implicit, true),
        b"Orya" => ('\u{B4D}', RephPlace::AfterMain, RephMode::Implicit, true),
        b"Taml" => ('\u{BCD}', RephPlace::AfterPost, RephMode::Implicit, true),
        b"Telu" => ('\u{C4D}', RephPlace::AfterPost, RephMode::Explicit, false),
        b"Knda" => ('\u{CCD}', RephPlace::AfterPost, RephMode::Implicit, false),
        b"Mlym" => ('\u{D4D}', RephPlace::AfterMain, RephMode::Logical, true),
        _ => ('\u{94D}', RephPlace::BeforePost, RephMode::Implicit, true),
    };
    Config {
        virama,
        reph_place,
        reph_mode,
        below_before,
    }
}

/// The kind and first place of `character`, from its Unicode syllabic and
/// positional categories.
fn describe(character: char, script: Script) -> (u8, u8) {
    let code = u32::from(character);
    let props = props(character);
    let kind = match code {
        0x200C => kind::NON_JOINER,
        0x200D => kind::JOINER,
        0x25CC => kind::DOTTED_CIRCLE,
        0x00A0 | 0x00D7 | 0x2012..=0x2015 | 0x2022 | 0x25FB..=0x25FE => kind::PLACEHOLDER,
        0x0930 | 0x09B0 | 0x09F0 | 0x0A30 | 0x0AB0 | 0x0B30 | 0x0BB0 | 0x0C30 | 0x0CB0 | 0x0D30 => {
            kind::RA
        }
        0x0951 | 0x0952 | 0x1CD0..=0x1CE8 | 0x1CED | 0x1CF4 | 0x1CF7..=0x1CF9 | 0xA8E0..=0xA8F1 => {
            kind::ACCENT
        }
        0x0A72 | 0x0A73 | 0x1CF5 | 0x1CF6 => kind::CONSONANT,
        0x0A51 => kind::MATRA,
        _ => match props.syllabic() {
            1 => kind::SYMBOL,
            2 | 18 | 28 | 32 => kind::MODIFIER,
            3 | 12 | 24 | 25 => kind::PLACEHOLDER,
            4 => kind::ACCENT,
            5 | 6 | 8 => kind::CONSONANT,
            7 | 11 | 15 | 16 => kind::MEDIAL,
            10 | 26 | 34 => kind::MATRA,
            13 => kind::REPHA,
            17 => kind::STACKER,
            19 | 31 => kind::HALANT,
            23 | 30 => kind::NUKTA,
            33 | 35 => kind::VOWEL,
            _ => kind::OTHER,
        },
    };
    let place = if bit(kind) & CONSONANTS != 0 {
        place::BASE
    } else if kind == kind::MATRA {
        matra_place(code, props.positional(), script)
    } else if bit(kind) & (bit(kind::MODIFIER) | bit(kind::ACCENT) | bit(kind::SYMBOL)) != 0 {
        place::MODIFIER
    } else {
        place::END
    };
    (kind, place)
}

/// Where a vowel sign goes, from the side of the consonant it is on.
fn matra_place(code: u32, side: u8, script: Script) -> u8 {
    let script = &script.0;
    match side {
        // Left, or a sign with a part on the left.
        2 | 4 | 5 | 11 | 12 | 14 | 15 => place::PRE_MATRA,
        // Right.
        3 | 7 | 10 | 13 => match script {
            b"Deva" => place::AFTER_SUB,
            b"Telu" if code <= 0x0C42 => place::BEFORE_SUB,
            b"Telu" => place::AFTER_SUB,
            b"Knda" if !(0x0CC3..=0x0CD6).contains(&code) => place::BEFORE_SUB,
            b"Knda" => place::AFTER_SUB,
            _ => place::AFTER_POST,
        },
        // Top.
        8 | 9 => match script {
            b"Guru" => place::AFTER_POST,
            b"Orya" => place::AFTER_MAIN,
            b"Telu" | b"Knda" => place::BEFORE_SUB,
            _ => place::AFTER_SUB,
        },
        // Bottom.
        1 => match script {
            b"Guru" | b"Gujr" | b"Taml" | b"Mlym" => place::AFTER_POST,
            b"Telu" | b"Knda" => place::BEFORE_SUB,
            _ => place::AFTER_SUB,
        },
        _ => place::AFTER_SUB,
    }
}

fn is(glyph: &Glyph, kinds: u32) -> bool {
    // A glyph made of several no longer counts as what one of them was.
    !glyph.ligated && bit(glyph.kind) & kinds != 0
}

fn is_consonant(glyph: &Glyph) -> bool {
    is(glyph, CONSONANTS)
}

fn is_joiner(glyph: &Glyph) -> bool {
    is(glyph, JOINERS)
}

fn is_halant(glyph: &Glyph) -> bool {
    is(glyph, bit(kind::HALANT))
}

/// The kinds of syllable, each with its pattern, in the order of choice
/// between two that match the same text.
fn grammar() -> Vec<(u8, Pattern)> {
    use kind::{
        ACCENT, CONSONANT, DOTTED_CIRCLE, HALANT, JOINER, MATRA, MEDIAL, MODIFIER, NON_JOINER,
        NUKTA, PLACEHOLDER, RA, REPHA, STACKER, SYMBOL, VOWEL,
    };
    let nuktas = || maybe(seq(vec![one(&[NUKTA]), maybe(one(&[NUKTA]))]));
    let joiner = || one(&[JOINER, NON_JOINER]);
    let consonant = || seq(vec![one(&[CONSONANT, RA]), maybe(one(&[JOINER])), nuktas()]);
    let reph = || Pattern::Either(vec![seq(vec![one(&[RA]), one(&[HALANT])]), one(&[REPHA])]);
    let halant_group = || {
        seq(vec![
            maybe(joiner()),
            one(&[HALANT]),
            maybe(seq(vec![one(&[JOINER]), maybe(one(&[NUKTA]))])),
        ])
    };
    let matra_group = || {
        seq(vec![
            repeat(joiner()),
            one(&[MATRA]),
            maybe(one(&[NUKTA])),
            maybe(one(&[HALANT])),
        ])
    };
    let tail = || {
        seq(vec![
            maybe(seq(vec![
                maybe(joiner()),
                one(&[MODIFIER]),
                maybe(one(&[MODIFIER])),
                maybe(one(&[NON_JOINER])),
            ])),
            repeat(one(&[ACCENT])),
        ])
    };
    let complex_tail = || {
        seq(vec![
            repeat(seq(vec![halant_group(), consonant()])),
            maybe(one(&[MEDIAL])),
            Pattern::Either(vec![
                halant_group(),
                seq(vec![one(&[HALANT]), one(&[NON_JOINER])]),
                repeat(matra_group()),
            ]),
            tail(),
        ])
    };
    vec![
        (
            syllable::CONSONANT,
            seq(vec![
                maybe(one(&[REPHA, STACKER])),
                consonant(),
                complex_tail(),
            ]),
        ),
        (
            syllable::VOWEL,
            seq(vec![
                maybe(reph()),
                one(&[VOWEL]),
                nuktas(),
                Pattern::Either(vec![one(&[JOINER]), complex_tail()]),
            ]),
        ),
        (
            syllable::STANDALONE,
            seq(vec![
                Pattern::Either(vec![
                    seq(vec![maybe(one(&[REPHA, STACKER])), one(&[PLACEHOLDER])]),
                    seq(vec![maybe(reph()), one(&[DOTTED_CIRCLE])]),
                ]),
                nuktas(),
                complex_tail(),
            ]),
        ),
        (
            syllable::SYMBOL,
            seq(vec![one(&[SYMBOL]), maybe(one(&[NUKTA])), tail()]),
        ),
        (
            syllable::BROKEN,
            seq(vec![maybe(reph()), nuktas(), complex_tail()]),
        ),
    ]
}

/// Number each glyph's syllable: the high bits a count that changes from
/// one syllable to the next, the low bits its kind.
fn find_syllables(glyphs: &mut [Glyph]) -> bool {
    let kinds: Vec<u8> = glyphs.iter().map(|glyph| glyph.kind).collect();
    let grammar = grammar();
    let mut broken = false;
    let mut serial = 1_u8;
    let mut start = 0;
    while start < kinds.len() {
        // The longest match wins, and the first listed among equals.
        let mut best = (start + 1, syllable::OTHER);
        for (kind, pattern) in &grammar {
            if let Some(end) = pattern.ends(&kinds, start).last().copied()
                && end > start
                && (end > best.0 || best.1 == syllable::OTHER)
            {
                best = (end, *kind);
            }
        }
        let (end, kind) = best;
        broken |= kind == syllable::BROKEN;
        for glyph in &mut glyphs[start..end] {
            glyph.syllable = (serial << 4) | kind;
        }
        serial = if serial == 15 { 1 } else { serial + 1 };
        start = end;
    }
    broken
}

/// The lookups of each feature the reordering asks questions of.
struct Forms {
    rphf: Vec<u16>,
    pref: Vec<u16>,
    blwf: Vec<u16>,
    pstf: Vec<u16>,
    vatu: Vec<u16>,
}

struct Plan<'f, 'a> {
    config: Config,
    script: Script,
    /// The font names the script by its older tag, whose forms are
    /// written consonant then virama.
    old_spec: bool,
    virama: Option<u16>,
    table: Option<&'f LayoutTable<'a>>,
    forms: Forms,
}

impl Plan<'_, '_> {
    /// Whether a feature's lookups would act on exactly `glyphs`.
    fn would(&self, feature: &[u16], glyphs: &[u16]) -> bool {
        let Some(table) = self.table else {
            return false;
        };
        // Old fonts look at what is around; new ones match the bare pair.
        let bare = !self.old_spec && self.script != Script(*b"Mlym");
        feature.iter().any(|index| {
            let Ok(lookup) = table.lookup(*index) else {
                return false;
            };
            lookup.subtables.iter().any(|data| {
                Substitution::read(lookup.kind, data)
                    .is_ok_and(|subtable| would_apply(&subtable, glyphs, bare))
            })
        })
    }
}

fn would_apply(subtable: &Substitution<'_>, glyphs: &[u16], bare: bool) -> bool {
    let Some((first, rest)) = glyphs.split_first() else {
        return false;
    };
    match subtable {
        Substitution::Single(single) => {
            rest.is_empty() && matches!(single.substitute(*first), Ok(Some(_)))
        }
        Substitution::Multiple(lists) | Substitution::Alternate(lists) => {
            rest.is_empty() && matches!(lists.glyphs(*first), Ok(Some(_)))
        }
        Substitution::Ligature(ligatures) => ligatures.starting(*first).is_ok_and(|made| {
            made.iter()
                .any(|ligature| ligature.rest.iter().eq(rest.iter().copied()))
        }),
        Substitution::Context(context) => context.rules(*first).is_ok_and(|rules| {
            rules.iter().any(|rule| {
                rule.input_len() == rest.len()
                    && (!bare || (rule.backtrack_len() == 0 && rule.lookahead_len() == 0))
                    && rest
                        .iter()
                        .enumerate()
                        .all(|(place, glyph)| rule.input_matches(place, *glyph).unwrap_or(false))
            })
        }),
        Substitution::Reverse(_) => false,
    }
}

/// The lookups feature `tag` runs for the script.
fn feature_lookups(table: &LayoutTable<'_>, scripts: &[Tag], tag: &[u8; 4]) -> Vec<u16> {
    lookups(table, scripts, &[Feature::every(tag)])
        .map(|found| found.iter().map(|lookup| lookup.index).collect())
        .unwrap_or_default()
}

/// Take a character apart or not, for these scripts: a few letters stay
/// whole whatever Unicode says.
pub(crate) fn keeps_whole(character: char) -> bool {
    matches!(character, '\u{931}' | '\u{9DC}' | '\u{9DD}' | '\u{B94}')
}

/// Compose two characters, for these scripts: a vowel sign in two parts
/// stays in two parts, and one Bengali letter Unicode keeps apart is put
/// together.
pub(crate) fn compose(first: char, second: char) -> Option<char> {
    if props(first).is_mark() {
        return None;
    }
    if (first, second) == ('\u{9AF}', '\u{9BC}') {
        return Some('\u{9DF}');
    }
    crate::unicode::compose(first, second)
}

/// Set each glyph's kind and first place.
pub(crate) fn describe_all(glyphs: &mut [Glyph], script: Script) {
    for glyph in glyphs {
        (glyph.kind, glyph.place) = describe(glyph.character, script);
    }
}

/// Run the substitutions for a run of one of these scripts: find the
/// syllables, reorder, apply the forms a feature at a time, reorder
/// again, and apply the rest.
pub(crate) fn substitute<'a>(
    font: &Font<'a>,
    mut engine: Option<Engine<'_, 'a>>,
    buffer: &mut Buffer,
    scripts: &[Tag],
    script: Script,
) -> Result<(), TextError> {
    let table = font.gsub();
    let chosen = table.and_then(|table| table.language(scripts, None).ok()?.script);
    let form =
        |tag: &[u8; 4]| table.map_or_else(Vec::new, |table| feature_lookups(table, scripts, tag));
    let config = config(script);
    let plan = Plan {
        config,
        script,
        old_spec: chosen.is_none_or(|tag| tag[3] != b'2'),
        virama: font.glyph_id(config.virama).ok().flatten(),
        table,
        forms: Forms {
            rphf: form(b"rphf"),
            pref: form(b"pref"),
            blwf: form(b"blwf"),
            pstf: form(b"pstf"),
            vatu: form(b"vatu"),
        },
    };
    let broken = find_syllables(buffer.glyphs_mut());
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
    let every = |tag: &[u8; 4]| Feature::every(tag).manual();
    let masked = |tag: &[u8; 4], mask: u32| Feature::masked(tag, mask).manual();
    stage(buffer, &[Feature::every(b"locl"), Feature::every(b"ccmp")])?;
    {
        let glyphs = buffer.glyphs_mut();
        consonant_places(&plan, glyphs);
        if broken {
            insert_dotted_circles(font, glyphs);
        }
        each_syllable(glyphs, |glyphs, start, end| {
            first_reordering(&plan, glyphs, start, end);
        });
    }
    for features in [
        [every(b"nukt")],
        [every(b"akhn")],
        [masked(b"rphf", RPHF)],
        [every(b"rkrf")],
        [masked(b"pref", PREF)],
        [masked(b"blwf", BLWF)],
        [masked(b"abvf", ABVF)],
        [masked(b"half", HALF)],
        [masked(b"pstf", PSTF)],
        [every(b"vatu")],
        [every(b"cjct")],
    ] {
        stage(buffer, &features)?;
    }
    {
        let glyphs = buffer.glyphs_mut();
        each_syllable(glyphs, |glyphs, start, end| {
            last_reordering(&plan, glyphs, start, end);
        });
    }
    // The forms that span a syllable, then what every script shares.
    let Some(engine) = engine.as_mut() else {
        return Ok(());
    };
    let Some(table) = table else {
        return Ok(());
    };
    let finishing = [
        masked(b"init", INIT),
        every(b"pres"),
        every(b"abvs"),
        every(b"blws"),
        every(b"psts"),
        every(b"haln"),
    ];
    let shared = [
        Feature::every(b"rlig"),
        Feature::every(b"calt"),
        Feature::every(b"rclt"),
        Feature::every(b"clig"),
    ];
    let mut last = lookups(table, scripts, &finishing)?;
    for lookup in &mut last {
        lookup.per_syllable = true;
    }
    for lookup in lookups(table, scripts, &shared)? {
        if !last.iter().any(|seen| seen.index == lookup.index) {
            last.push(lookup);
        }
    }
    last.sort_by_key(|lookup| lookup.index);
    for lookup in &last {
        engine.apply(buffer, lookup)?;
    }
    Ok(())
}

/// Whether each consonant goes below or behind the main one, by asking
/// the font what it would do with the consonant and a virama.
fn consonant_places(plan: &Plan<'_, '_>, glyphs: &mut [Glyph]) {
    let Some(virama) = plan.virama else {
        return;
    };
    for glyph in glyphs {
        if glyph.place != place::BASE {
            continue;
        }
        let before = [virama, glyph.id];
        let after = [glyph.id, virama];
        let would = |feature: &[u16]| plan.would(feature, &before) || plan.would(feature, &after);
        glyph.place = if would(&plan.forms.blwf) || would(&plan.forms.vatu) {
            place::BELOW
        } else if would(&plan.forms.pstf) || would(&plan.forms.pref) {
            place::POST
        } else {
            place::BASE
        };
    }
}

/// A syllable with no consonant to carry its signs gets a dotted circle
/// to carry them, when the font has one.
fn insert_dotted_circles(font: &Font<'_>, glyphs: &mut Vec<Glyph>) {
    let Some(circle) = font.glyph_id('\u{25CC}').ok().flatten() else {
        return;
    };
    let mut index = 0;
    let mut last = 0;
    while index < glyphs.len() {
        let syllable = glyphs[index].syllable;
        if syllable == last || syllable & 0x0F != syllable::BROKEN {
            last = syllable;
            index += 1;
            continue;
        }
        last = syllable;
        // After a hook written as its own character.
        let mut at = index;
        while at < glyphs.len() && glyphs[at].syllable == syllable && glyphs[at].kind == kind::REPHA
        {
            at += 1;
        }
        let mut glyph = Glyph::new('\u{25CC}', glyphs[index].cluster);
        glyph.id = circle;
        glyph.kind = kind::DOTTED_CIRCLE;
        glyph.place = place::END;
        glyph.syllable = syllable;
        glyph.mask = glyphs[index].mask;
        glyph.class = font
            .gdef()
            .filter(|gdef| gdef.has_classes())
            .map_or(GlyphClass::Base, |gdef| gdef.class(circle));
        glyph.category = gc::OTHER_LETTER;
        glyphs.insert(at, glyph);
        index = at + 1;
    }
}

/// Put a syllable's characters in the order they are written, before the
/// font's forms are applied.
#[allow(clippy::too_many_lines)]
fn first_reordering(plan: &Plan<'_, '_>, glyphs: &mut [Glyph], start: usize, end: usize) {
    if glyphs[start].syllable & 0x0F > syllable::STANDALONE
        && glyphs[start].syllable & 0x0F != syllable::BROKEN
    {
        return;
    }
    let config = plan.config;
    // Kannada: r, virama, joiner reads as r, joiner, virama.
    if plan.script == Script(*b"Knda")
        && start + 3 <= end
        && is(&glyphs[start], bit(kind::RA))
        && is(&glyphs[start + 1], bit(kind::HALANT))
        && is(&glyphs[start + 2], bit(kind::JOINER))
    {
        merge(glyphs, start + 1, start + 3);
        glyphs.swap(start + 1, start + 2);
    }
    // Find the main consonant: the last that has no form below or behind.
    let mut base = end;
    let mut has_reph = false;
    let mut limit = start;
    if !plan.forms.rphf.is_empty()
        && start + 3 <= end
        && ((config.reph_mode == RephMode::Implicit && !is_joiner(&glyphs[start + 2]))
            || (config.reph_mode == RephMode::Explicit && glyphs[start + 2].kind == kind::JOINER))
    {
        let pair = [glyphs[start].id, glyphs[start + 1].id];
        let triple = [pair[0], pair[1], glyphs[start + 2].id];
        if plan.would(&plan.forms.rphf, &pair)
            || (config.reph_mode == RephMode::Explicit && plan.would(&plan.forms.rphf, &triple))
        {
            limit += 2;
            while limit < end && is_joiner(&glyphs[limit]) {
                limit += 1;
            }
            base = start;
            has_reph = true;
        }
    } else if config.reph_mode == RephMode::Logical && glyphs[start].kind == kind::REPHA {
        limit += 1;
        while limit < end && is_joiner(&glyphs[limit]) {
            limit += 1;
        }
        base = start;
        has_reph = true;
    }
    {
        let mut index = end;
        let mut seen_below = false;
        while index > limit {
            index -= 1;
            if is_consonant(&glyphs[index]) {
                let place = glyphs[index].place;
                if place != place::BELOW && (place != place::POST || seen_below) {
                    base = index;
                    break;
                }
                if place == place::BELOW {
                    seen_below = true;
                }
                base = index;
            } else if start < index
                && glyphs[index].kind == kind::JOINER
                && glyphs[index - 1].kind == kind::HALANT
            {
                // A joiner after a virama asks for a half form: stop.
                break;
            }
        }
    }
    // With no other consonant the r is the main one, and no hook is made.
    if has_reph && base == start && limit - base <= 2 {
        has_reph = false;
    }
    for glyph in &mut glyphs[start..base.min(end)] {
        glyph.place = glyph.place.min(place::PRE_CONSONANT);
    }
    if base < end {
        glyphs[base].place = place::BASE;
    }
    if has_reph {
        glyphs[start].place = place::RA_TO_REPH;
    }
    // An old font writes a form as consonant then virama: move the first
    // virama after the main consonant to after the last consonant.
    if plan.old_spec {
        let double = plan.script == Script(*b"Knda");
        for index in base + 1..end {
            if glyphs[index].kind != kind::HALANT {
                continue;
            }
            let mut last = end - 1;
            while last > index
                && !(is_consonant(&glyphs[last]) || (double && glyphs[last].kind == kind::HALANT))
            {
                last -= 1;
            }
            if glyphs[last].kind != kind::HALANT && last > index {
                glyphs[index..=last].rotate_left(1);
            }
            break;
        }
    }
    // Marks go with the character before them.
    let mut last_place = place::START;
    for index in start..end {
        let kinds = JOINERS | bit(kind::NUKTA) | bit(kind::MEDIAL) | bit(kind::HALANT);
        if bit(glyphs[index].kind) & kinds != 0 {
            glyphs[index].place = last_place;
            if glyphs[index].kind == kind::HALANT && glyphs[index].place == place::PRE_MATRA {
                // A virama does not move with a vowel sign on the left.
                for before in (start..index).rev() {
                    if glyphs[before].place != place::PRE_MATRA {
                        glyphs[index].place = glyphs[before].place;
                        break;
                    }
                }
            }
        } else if glyphs[index].place != place::MODIFIER {
            last_place = glyphs[index].place;
        }
    }
    // A consonant after the main one owns what is written before it.
    let mut last = base;
    for index in base + 1..end {
        if is_consonant(&glyphs[index]) {
            let place = glyphs[index].place;
            for glyph in &mut glyphs[last + 1..index] {
                if glyph.place < place::MODIFIER {
                    glyph.place = place;
                }
            }
            last = index;
        } else if glyphs[index].kind == kind::MATRA {
            last = index;
        }
    }
    // Sort by place, keeping the order of equals, and remember where
    // each glyph came from.
    let mut order: Vec<usize> = (start..end).collect();
    order.sort_by_key(|index| glyphs[*index].place);
    let sorted: Vec<Glyph> = order.iter().map(|index| glyphs[*index].clone()).collect();
    glyphs[start..end].clone_from_slice(&sorted);
    // Find the main consonant again, and turn a run of left vowel signs
    // around, each keeping its own marks after it.
    let mut left: Option<(usize, usize)> = None;
    base = end;
    for (index, glyph) in glyphs.iter().enumerate().take(end).skip(start) {
        if glyph.place == place::BASE {
            base = index;
            break;
        }
        if glyph.place == place::PRE_MATRA {
            left = Some((left.map_or(index, |(first, _)| first), index));
        }
    }
    if let Some((first, last)) = left.filter(|(first, last)| first < last) {
        glyphs[first..=last].reverse();
        let mut from = first;
        for index in first..=last {
            if glyphs[index].kind == kind::MATRA {
                glyphs[from..=index].reverse();
                from = index + 1;
            }
        }
    }
    // What moved after the main consonant now shares a cluster with what
    // it moved past.
    if plan.old_spec || end - start > 127 {
        merge(glyphs, base, end);
    } else {
        let mut seen = vec![false; end - start];
        for at in base..end {
            if seen[at - start] {
                continue;
            }
            let (mut low, mut high) = (at, at);
            let mut from = order[at - start];
            while from != at {
                low = low.min(from);
                high = high.max(from);
                seen[from - start] = true;
                from = order[from - start];
            }
            merge(glyphs, base.max(low), high + 1);
        }
    }
    // Say which forms apply where.
    for glyph in &mut glyphs[start..end] {
        if glyph.place != place::RA_TO_REPH {
            break;
        }
        glyph.mask |= RPHF;
    }
    let mut before = HALF;
    if !plan.old_spec && config.below_before {
        before |= BLWF;
    }
    for glyph in &mut glyphs[start..base.min(end)] {
        glyph.mask |= before;
    }
    for glyph in glyphs[..end].iter_mut().skip(base + 1) {
        glyph.mask |= BLWF | ABVF | PSTF;
    }
    // An old Devanagari font draws r before the main consonant as an
    // eyelash, through the below-base form.
    if plan.old_spec && plan.script == Script(*b"Deva") {
        let mut index = start;
        while index + 1 < base {
            if glyphs[index].kind == kind::RA
                && glyphs[index + 1].kind == kind::HALANT
                && (index + 2 == base || glyphs[index + 2].kind != kind::JOINER)
            {
                glyphs[index].mask |= BLWF;
                glyphs[index + 1].mask |= BLWF;
            }
            index += 1;
        }
    }
    // A virama and r after the main consonant that the font draws before
    // it are marked for that.
    if !plan.forms.pref.is_empty() && base + 2 < end {
        for index in base + 1..end - 1 {
            if plan.would(&plan.forms.pref, &[glyphs[index].id, glyphs[index + 1].id]) {
                glyphs[index].mask |= PREF;
                glyphs[index + 1].mask |= PREF;
                break;
            }
        }
    }
    // A non-joiner stops the half form of the consonant before it.
    for index in start + 1..end {
        if !is_joiner(&glyphs[index]) {
            continue;
        }
        let non_joiner = glyphs[index].kind == kind::NON_JOINER;
        let mut before = index;
        loop {
            before -= 1;
            if non_joiner {
                glyphs[before].mask &= !HALF;
            }
            if before <= start || is_consonant(&glyphs[before]) {
                break;
            }
        }
    }
}

/// Put what the font's forms made in its last place: a vowel sign on the
/// left nearer the main consonant, the hook of an initial `r` after it.
#[allow(clippy::too_many_lines)]
fn last_reordering(plan: &Plan<'_, '_>, glyphs: &mut [Glyph], start: usize, end: usize) {
    let config = plan.config;
    // A virama a lookup both joined and split is a virama again.
    if let Some(virama) = plan.virama {
        for glyph in &mut glyphs[start..end] {
            if glyph.id == virama && glyph.ligated && glyph.multiplied {
                glyph.kind = kind::HALANT;
                glyph.ligated = false;
                glyph.multiplied = false;
            }
        }
    }
    let formed = |glyph: &Glyph| glyph.ligated && !glyph.multiplied;
    let mut try_pref = !plan.forms.pref.is_empty();
    // Find the main consonant again.
    let mut base = start;
    while base < end {
        if glyphs[base].place >= place::BASE {
            if try_pref && base + 1 < end {
                for index in base + 1..end {
                    if glyphs[index].mask & PREF == 0 {
                        continue;
                    }
                    if !(glyphs[index].substituted && formed(&glyphs[index])) {
                        // The form was not made: the main consonant is here.
                        base = index;
                        while base < end && is_halant(&glyphs[base]) {
                            base += 1;
                        }
                        if base < end {
                            glyphs[base].place = place::BASE;
                        }
                        try_pref = false;
                    }
                    break;
                }
                if base == end {
                    break;
                }
            }
            // Malayalam: pass the below-base forms that were not made.
            if plan.script == Script(*b"Mlym") {
                let mut index = base + 1;
                while index < end {
                    while index < end && is_joiner(&glyphs[index]) {
                        index += 1;
                    }
                    if index == end || !is_halant(&glyphs[index]) {
                        break;
                    }
                    index += 1;
                    while index < end && is_joiner(&glyphs[index]) {
                        index += 1;
                    }
                    if index < end
                        && is_consonant(&glyphs[index])
                        && glyphs[index].place == place::BELOW
                    {
                        base = index;
                        glyphs[base].place = place::BASE;
                    }
                    index += 1;
                }
            }
            if start < base && glyphs[base].place > place::BASE {
                base -= 1;
            }
            break;
        }
        base += 1;
    }
    if base == end && start < base && is(&glyphs[base - 1], bit(kind::JOINER)) {
        base -= 1;
    }
    if base < end {
        while start < base && is(&glyphs[base], bit(kind::NUKTA) | bit(kind::HALANT)) {
            base -= 1;
        }
    }
    let halves = plan.script != Script(*b"Mlym") && plan.script != Script(*b"Taml");
    // A vowel sign on the left goes after the last half form that kept
    // its virama, to sit next to the main consonant.
    if start + 1 < end && start < base {
        let mut new_place = if base == end { base - 2 } else { base - 1 };
        if halves {
            loop {
                while new_place > start
                    && !is(&glyphs[new_place], bit(kind::MATRA) | bit(kind::HALANT))
                {
                    new_place -= 1;
                }
                if is_halant(&glyphs[new_place]) && glyphs[new_place].place != place::PRE_MATRA {
                    // A joiner after the virama: look further back.
                    if new_place + 1 < end
                        && glyphs[new_place + 1].kind == kind::JOINER
                        && new_place > start
                    {
                        new_place -= 1;
                        continue;
                    }
                } else {
                    new_place = start;
                }
                break;
            }
        }
        if start < new_place && glyphs[new_place].place != place::PRE_MATRA {
            let mut index = new_place;
            while index > start {
                if glyphs[index - 1].place == place::PRE_MATRA {
                    let old = index - 1;
                    if old < base && base <= new_place {
                        base -= 1;
                    }
                    glyphs[old..=new_place].rotate_left(1);
                    merge(glyphs, new_place, end.min(base + 1));
                    new_place -= 1;
                }
                index -= 1;
            }
        } else {
            for index in start..base {
                if glyphs[index].place == place::PRE_MATRA {
                    merge(glyphs, index, end.min(base + 1));
                    break;
                }
            }
        }
    }
    // The hook of an initial r: moved when the font made it, or when it
    // was written as its own character.
    if start + 1 < end
        && glyphs[start].place == place::RA_TO_REPH
        && ((glyphs[start].kind == kind::REPHA) ^ formed(&glyphs[start]))
    {
        // After the first virama that was kept between the hook and the
        // main consonant.
        let after_halant = |glyphs: &[Glyph]| {
            let mut at = start + 1;
            while at < base && !is_halant(&glyphs[at]) {
                at += 1;
            }
            (at < base && is_halant(&glyphs[at])).then(|| {
                if at + 1 < base && is_joiner(&glyphs[at + 1]) {
                    at + 1
                } else {
                    at
                }
            })
        };
        let mut target = None;
        if config.reph_place != RephPlace::AfterPost {
            target = after_halant(glyphs);
            if target.is_none() && config.reph_place == RephPlace::AfterMain {
                let mut at = base;
                while at + 1 < end && glyphs[at + 1].place <= place::AFTER_MAIN {
                    at += 1;
                }
                target = (at < end).then_some(at);
            }
            if target.is_none() && config.reph_place == RephPlace::AfterSub {
                let mut at = base;
                while at + 1 < end
                    && !matches!(
                        glyphs[at + 1].place,
                        place::POST | place::AFTER_POST | place::MODIFIER
                    )
                {
                    at += 1;
                }
                target = (at < end).then_some(at);
            }
        }
        let target = target.or_else(|| after_halant(glyphs)).unwrap_or_else(|| {
            // The end of the syllable, before its modifiers.
            let mut at = end - 1;
            while at > start && glyphs[at].place == place::MODIFIER {
                at -= 1;
            }
            // Before a virama that follows a vowel sign, so the hook sits
            // on the sign.
            if is_halant(&glyphs[at]) {
                let signs = (base + 1..at)
                    .filter(|index| glyphs[*index].kind == kind::MATRA)
                    .count();
                at -= signs.min(at - start);
            }
            at
        });
        merge(glyphs, start, target + 1);
        glyphs[start..=target].rotate_left(1);
        if start < base && base <= target {
            base -= 1;
        }
    }
    // An r the font drew before the main consonant goes where a left
    // vowel sign would.
    if try_pref && base + 1 < end {
        for index in base + 1..end {
            if glyphs[index].mask & PREF == 0 {
                continue;
            }
            if formed(&glyphs[index]) {
                let mut new_place = base;
                if halves {
                    while new_place > start
                        && !is(&glyphs[new_place - 1], bit(kind::MATRA) | bit(kind::HALANT))
                    {
                        new_place -= 1;
                    }
                }
                if new_place > start
                    && is_halant(&glyphs[new_place - 1])
                    && new_place < end
                    && is_joiner(&glyphs[new_place])
                {
                    new_place += 1;
                }
                merge(glyphs, new_place, index + 1);
                glyphs[new_place..=index].rotate_right(1);
            }
            break;
        }
    }
    // A left vowel sign that starts a word may take a form of its own.
    if glyphs[start].place == place::PRE_MATRA {
        // Not after a letter, a mark, or a character that is not drawn.
        let word_start = start == 0
            || !matches!(
                glyphs[start - 1].category,
                gc::UNASSIGNED..=gc::SPACING_MARK | gc::FORMAT..=gc::SURROGATE
            );
        if word_start {
            glyphs[start].mask |= INIT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{describe, find_syllables, kind, place, syllable};
    use crate::script::Script;
    use crate::shaper::buffer::Glyph;

    const DEVA: Script = Script(*b"Deva");

    /// The syllables of `text`: each one's characters and its kind.
    fn syllables(text: &str) -> Vec<(String, u8)> {
        let mut glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| {
                let mut glyph = Glyph::new(character, at);
                (glyph.kind, glyph.place) = describe(character, DEVA);
                glyph
            })
            .collect();
        let _broken = find_syllables(&mut glyphs);
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

    #[test]
    fn characters_have_their_kind_and_place() {
        assert_eq!(describe('\u{915}', DEVA), (kind::CONSONANT, place::BASE));
        assert_eq!(describe('\u{930}', DEVA), (kind::RA, place::BASE));
        assert_eq!(describe('\u{94D}', DEVA).0, kind::HALANT);
        assert_eq!(describe('\u{93F}', DEVA), (kind::MATRA, place::PRE_MATRA));
        assert_eq!(describe('\u{940}', DEVA), (kind::MATRA, place::AFTER_SUB));
        assert_eq!(describe('\u{947}', DEVA), (kind::MATRA, place::AFTER_SUB));
        assert_eq!(describe('\u{902}', DEVA), (kind::MODIFIER, place::MODIFIER));
        assert_eq!(describe('\u{93C}', DEVA).0, kind::NUKTA);
        assert_eq!(describe('\u{905}', DEVA).0, kind::VOWEL);
        assert_eq!(describe('\u{200D}', DEVA).0, kind::JOINER);
        assert_eq!(describe('\u{25CC}', DEVA).0, kind::DOTTED_CIRCLE);
        assert_eq!(describe('a', DEVA).0, kind::OTHER);
        // Bengali e and its two-part vowel signs go on the left.
        assert_eq!(describe('\u{9C7}', Script(*b"Beng")).1, place::PRE_MATRA);
        assert_eq!(describe('\u{9BE}', Script(*b"Beng")).1, place::AFTER_POST);
    }

    #[test]
    fn text_is_cut_into_syllables() {
        // namaste: na, ma, s-te.
        assert_eq!(
            syllables("\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947}"),
            [
                ("\u{928}".to_owned(), syllable::CONSONANT),
                ("\u{92E}".to_owned(), syllable::CONSONANT),
                (
                    "\u{938}\u{94D}\u{924}\u{947}".to_owned(),
                    syllable::CONSONANT
                ),
            ]
        );
        // A vowel with a modifier, then a letter with a left vowel sign.
        assert_eq!(
            syllables("\u{905}\u{902}\u{915}\u{93F}"),
            [
                ("\u{905}\u{902}".to_owned(), syllable::VOWEL),
                ("\u{915}\u{93F}".to_owned(), syllable::CONSONANT),
            ]
        );
        // A sign with no consonant is a broken syllable; Latin is other.
        assert_eq!(syllables("\u{93F}")[0].1, syllable::BROKEN);
        assert_eq!(
            syllables("a\u{915}"),
            [
                ("a".to_owned(), syllable::OTHER),
                ("\u{915}".to_owned(), syllable::CONSONANT),
            ]
        );
        // A final virama, and one a non-joiner ends.
        assert_eq!(syllables("\u{915}\u{94D}").len(), 1);
        assert_eq!(syllables("\u{915}\u{94D}\u{200C}\u{937}").len(), 2);
        assert_eq!(syllables("\u{915}\u{94D}\u{200D}\u{937}").len(), 1);
        assert!(syllables("").is_empty());
    }

    #[test]
    fn a_long_run_is_cut_in_time() {
        let started = std::time::Instant::now();
        for unit in [
            "\u{915}\u{94D}",
            "\u{915}",
            "\u{93F}",
            "\u{915}\u{93C}\u{93F}\u{902}",
        ] {
            let text = unit.repeat(20_000);
            assert!(!syllables(&text).is_empty());
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }
}

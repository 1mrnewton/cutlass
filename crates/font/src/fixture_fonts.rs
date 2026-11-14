//! The fonts under `fixtures/` that are built in code: what each holds,
//! and a check that the files are what this builder writes.
//!
//! To write them again after a change here:
//! `cargo test -p cutlass-font -- --ignored write_the_fixture_fonts`.

use crate::fixture::{
    Glyph, LookupSpec, boxed, chain, font, gdef, layout_table, ligatures, mark_attach, pairs,
    sequences, single,
};

/// A mark: no advance, its ink above the letter before it.
fn mark() -> Glyph {
    Glyph {
        advance: 0,
        outline: Some([-200, 750, -100, 850]),
    }
}

/// A face for shaping tests, with a few letters of three scripts.
///
/// Latin: `A` 2, `V` 3, `f` 4, `i` 5, the ligature `fi` 6, a combining
/// acute 7, `e` 8, `é` 9, `(` 31, `)` 32, `x` 34. `A` and `V` kern by
/// -100, and the acute sits on an anchor over `A` and `e`.
///
/// Arabic: beh 10 with its initial 11, medial 12 and final 13 forms; alef
/// 14 and its final form 15; lam 16 with 17, 18 and 19; the lam-alef
/// ligature 20; and a fatha 21 that sits over every form of beh.
///
/// Devanagari: ka 22, ssa 23, the virama 24, half ka 25, the conjunct
/// kssa 26, the vowel sign i 27, ra 28, the hook ra becomes 29, ta 30,
/// and the dotted circle 33.
pub(crate) fn shaping() -> Vec<u8> {
    let mut glyphs = vec![boxed(500); 35];
    for (glyph, advance) in [
        (1, 250),
        (2, 600),
        (3, 600),
        (4, 300),
        (5, 250),
        (10, 800),
        (11, 400),
        (12, 400),
        (13, 800),
        (14, 300),
        (15, 350),
        (16, 600),
        (17, 300),
        (18, 300),
        (19, 600),
        (20, 700),
        (22, 700),
        (23, 700),
        (25, 400),
        (26, 900),
        (27, 300),
        (30, 650),
        (31, 300),
        (32, 300),
        (33, 600),
    ] {
        glyphs[glyph] = boxed(advance);
    }
    glyphs[0].outline = None;
    glyphs[1].outline = None;
    for glyph in [7, 21, 24, 29] {
        glyphs[glyph] = mark();
    }
    let mut map = vec![
        (' ', 1),
        ('A', 2),
        ('V', 3),
        ('f', 4),
        ('i', 5),
        ('\u{301}', 7),
        ('e', 8),
        ('\u{E9}', 9),
        ('\u{628}', 10),
        ('\u{627}', 14),
        ('\u{644}', 16),
        ('\u{64E}', 21),
        ('\u{915}', 22),
        ('\u{937}', 23),
        ('\u{94D}', 24),
        ('\u{93F}', 27),
        ('\u{930}', 28),
        ('\u{924}', 30),
        ('(', 31),
        (')', 32),
        ('\u{25CC}', 33),
        ('x', 34),
    ];
    map.sort_unstable();
    let substitutions = layout_table(
        &[
            (b"arab", &[2, 3, 4, 5]),
            (b"dev2", &[0, 6, 7]),
            (b"latn", &[1]),
        ],
        &[
            (b"akhn", &[5]),
            (b"liga", &[0]),
            (b"fina", &[3]),
            (b"init", &[1]),
            (b"medi", &[2]),
            (b"rlig", &[4]),
            (b"half", &[7]),
            (b"rphf", &[6]),
        ],
        vec![
            LookupSpec::new(4, vec![ligatures(&[(4, &[(&[5], 6)])])]),
            LookupSpec::new(1, vec![single(&[(10, 11), (16, 17)])]),
            LookupSpec::new(1, vec![single(&[(10, 12), (16, 18)])]),
            LookupSpec::new(1, vec![single(&[(10, 13), (14, 15), (16, 19)])]),
            LookupSpec::new(4, vec![ligatures(&[(17, &[(&[15], 20)])])]).flag(8),
            LookupSpec::new(4, vec![ligatures(&[(22, &[(&[24, 23], 26)])])]),
            LookupSpec::new(4, vec![ligatures(&[(28, &[(&[24], 29)])])]),
            LookupSpec::new(4, vec![ligatures(&[(22, &[(&[24], 25)])])]),
        ],
    );
    let above = (300, 700);
    let positioning = layout_table(
        &[(b"arab", &[1]), (b"latn", &[0, 1])],
        &[(b"kern", &[0]), (b"mark", &[1])],
        vec![
            LookupSpec::new(2, vec![pairs(&[(2, &[(3, -100)])])]),
            LookupSpec::new(
                4,
                vec![mark_attach(
                    &[(7, 0, (-150, 750)), (21, 0, (-150, 750))],
                    &[
                        (2, &[above]),
                        (8, &[(250, 520)]),
                        (10, &[(400, 600)]),
                        (11, &[(200, 600)]),
                        (12, &[(200, 600)]),
                        (13, &[(400, 600)]),
                    ],
                )],
            ),
        ],
    );
    let kinds = gdef(
        &[
            (1, 5, 1),
            (6, 6, 2),
            (7, 7, 3),
            (8, 19, 1),
            (20, 20, 2),
            (21, 21, 3),
            (22, 23, 1),
            (24, 24, 3),
            (25, 25, 1),
            (26, 26, 2),
            (27, 28, 1),
            (29, 29, 3),
            (30, 34, 1),
        ],
        &[],
        &[],
    );
    font(
        &glyphs,
        &map,
        &[
            (b"GDEF", kinds),
            (b"GPOS", positioning),
            (b"GSUB", substitutions),
        ],
    )
}

/// The lookups of a face whose rules never settle: eight that each write
/// four glyphs for glyph 2, and two contextual ones on glyph 3 that call
/// each other.
fn restless() -> Vec<LookupSpec> {
    let mut lookups: Vec<LookupSpec> = (0..8)
        .map(|_| LookupSpec::new(2, vec![sequences(&[(2, &[2, 2, 2, 2])])]))
        .collect();
    lookups.push(LookupSpec::new(
        6,
        vec![chain(&[], &[&[3]], &[], &[(0, 9)])],
    ));
    lookups.push(LookupSpec::new(
        6,
        vec![chain(&[], &[&[3]], &[], &[(0, 8)])],
    ));
    lookups
}

/// A face whose rules never settle: eight lookups that each write four
/// glyphs for one, and two contextual lookups that call each other.
pub(crate) fn hostile() -> Vec<u8> {
    let glyphs = vec![boxed(500); 4];
    let substitutions = layout_table(
        &[(b"DFLT", &[0, 1])],
        &[(b"calt", &[8]), (b"ccmp", &[0, 1, 2, 3, 4, 5, 6, 7])],
        restless(),
    );
    font(&glyphs, &[('A', 2), ('B', 3)], &[(b"GSUB", substitutions)])
}

/// The same rules under the names of four scripts shaped by syllable,
/// Javanese, Khmer, Mongolian and Myanmar, in features each of them
/// applies: the first letter of each is the glyph that grows, and the
/// second the one whose lookups call each other.
pub(crate) fn hostile_syllables() -> Vec<u8> {
    let glyphs = vec![boxed(500); 4];
    let substitutions = layout_table(
        &[
            (b"java", &[0, 1]),
            (b"khmr", &[0, 1]),
            (b"mong", &[0, 1]),
            (b"mym2", &[0, 1]),
        ],
        &[(b"ccmp", &[0, 1, 2, 3, 4, 5, 6, 7]), (b"pres", &[8])],
        restless(),
    );
    font(
        &glyphs,
        &[
            ('\u{1000}', 2),
            ('\u{1001}', 3),
            ('\u{1780}', 2),
            ('\u{1781}', 3),
            ('\u{1820}', 2),
            ('\u{1821}', 3),
            ('\u{A98F}', 2),
            ('\u{A990}', 3),
        ],
        &[(b"GSUB", substitutions)],
    )
}

/// A face that shapes by state machines, as faces made for Apple's
/// layout engine do: `A` 2, `B` 3, the ligature of the two 4, and `D` 5.
/// Its one chain first swaps `D` for `B`, then makes `A` and `B` one
/// glyph.
pub(crate) fn machines() -> Vec<u8> {
    use crate::fixture::be;
    let long = |values: &[u32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect()
    };
    // A lookup of single glyphs: each a glyph and its value.
    let singles = |pairs: &[(u16, u16)]| {
        let mut table = be(&[6, 4, u16::try_from(pairs.len()).expect("count"), 0, 0, 0]);
        for (glyph, value) in pairs {
            table.extend(be(&[*glyph, *value]));
        }
        table
    };
    let subtable = |kind: u32, body: Vec<u8>| {
        let mut bytes = long(&[u32::try_from(body.len() + 12).expect("length"), kind, 1]);
        bytes.extend(body);
        bytes
    };
    // The ligature machine. Classes: 4 is A, 5 is B. States 0 and 1 wait
    // for an A; state 2 has seen one. Entry 1 marks an A; entry 2 marks a
    // B after it and joins the two.
    let classes = singles(&[(2, 4), (3, 5)]);
    let states = be(&[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2]);
    let entries = be(&[0, 0, 0, 2, 0x8000, 0, 0, 0xA000, 0]);
    // Popped last first: B is component 0 and A component 1; both add
    // nothing, so the ligature is the first in its list.
    let actions = long(&[0x3FFF_FFFD, 0xFFFF_FFFF]);
    let components = be(&[0, 0]);
    let ligatures = be(&[4]);
    let mut at = 28;
    let mut offsets = Vec::new();
    for part in [
        &classes,
        &states,
        &entries,
        &actions,
        &components,
        &ligatures,
    ] {
        offsets.push(u32::try_from(at).expect("offset"));
        at += part.len();
    }
    let mut ligature = long(&[6]);
    ligature.extend(long(&offsets));
    for part in [classes, states, entries, actions, components, ligatures] {
        ligature.extend(part);
    }
    let swap = subtable(4, singles(&[(5, 3)]));
    let join = subtable(2, ligature);
    let mut chain = long(&[
        1,
        u32::try_from(16 + swap.len() + join.len()).expect("length"),
        0,
        2,
    ]);
    chain.extend(swap);
    chain.extend(join);
    let mut morx = be(&[2, 0]);
    morx.extend(long(&[1]));
    morx.extend(chain);
    let glyphs = vec![boxed(500); 6];
    font(&glyphs, &[('A', 2), ('B', 3), ('D', 5)], &[(b"morx", morx)])
}

/// A face whose character map lists variation sequences, as an emoji face
/// and a face of Han characters do: `A` 1, `1` 2, the keycap `1` with the
/// emoji selector draws 3, a glyph for that selector 4, the `A` the text
/// selector picks 5, and the enclosing keycap 6. `A` with the emoji
/// selector is listed as drawing what `A` alone does.
pub(crate) fn varied() -> Vec<u8> {
    let glyphs = [
        boxed(500),
        boxed(600),
        boxed(500),
        boxed(900),
        boxed(100),
        boxed(650),
        boxed(800),
    ];
    let map = [('1', 2), ('A', 1), ('\u{20E3}', 6), ('\u{FE0F}', 4)];
    let pairs = [
        ('1', '\u{FE0F}', Some(3)),
        ('A', '\u{FE0F}', None),
        ('A', '\u{FE0E}', Some(5)),
    ];
    let cmap = crate::fixture::character_map(&map, &pairs);
    font(&glyphs, &map, &[(b"cmap", cmap)])
}

/// The Han character for water, which Japan, Korea and China each write
/// in their own way.
const WATER: char = '\u{6C34}';

/// A face of Han characters alone, as one made for Chinese: water is 1,
/// 500 wide.
pub(crate) fn han() -> Vec<u8> {
    font(&[boxed(500), boxed(500)], &[(WATER, 1)], &[])
}

/// A face as one made for Japanese: the kana `a` is 1 and water 2, 700
/// wide.
pub(crate) fn han_kana() -> Vec<u8> {
    let glyphs = [boxed(500), boxed(700), boxed(700)];
    font(&glyphs, &[('\u{3042}', 1), (WATER, 2)], &[])
}

/// A face as one made for Korean: water is 1, 900 wide, and the Hangul
/// syllable `ga` 2.
pub(crate) fn han_hangul() -> Vec<u8> {
    let glyphs = [boxed(500), boxed(900), boxed(900)];
    font(&glyphs, &[(WATER, 1), ('\u{AC00}', 2)], &[])
}

/// A face for Khmer: ka 1, ro 2, the coeng 3, the vowel e 4, the vowel
/// aa 5, the upper part of the vowel oe 6, the dotted circle 7, the sign
/// nikahit 8, ta 9, and a shifter 15.
///
/// The coeng and ta are the subscript ta 10; the coeng and ro are the ro
/// written before its letter, 11; a subscript ta after such a ro is the
/// narrow one 13; and ka with aa is one glyph, 14. The nikahit sits on an
/// anchor over ka and over that glyph.
pub(crate) fn khmer() -> Vec<u8> {
    let mut glyphs = vec![boxed(500); 16];
    for (glyph, advance) in [(1, 700), (2, 600), (4, 400), (5, 350), (7, 600), (9, 650)] {
        glyphs[glyph] = boxed(advance);
    }
    glyphs[11] = boxed(300);
    glyphs[14] = boxed(1_000);
    for glyph in [3, 6, 8, 10, 13, 15] {
        glyphs[glyph] = mark();
    }
    let mut map = vec![
        ('\u{1780}', 1),
        ('\u{179A}', 2),
        ('\u{17D2}', 3),
        ('\u{17C1}', 4),
        ('\u{17B6}', 5),
        ('\u{17BE}', 6),
        ('\u{25CC}', 7),
        ('\u{17C6}', 8),
        ('\u{178F}', 9),
        ('\u{17C9}', 15),
    ];
    map.sort_unstable();
    let substitutions = layout_table(
        &[(b"khmr", &[0, 1, 2, 3])],
        &[
            (b"blwf", &[1]),
            (b"cfar", &[2]),
            (b"pref", &[0]),
            (b"pres", &[3]),
        ],
        vec![
            LookupSpec::new(4, vec![ligatures(&[(3, &[(&[2], 11)])])]),
            LookupSpec::new(4, vec![ligatures(&[(3, &[(&[9], 10)])])]),
            LookupSpec::new(1, vec![single(&[(10, 13)])]),
            LookupSpec::new(4, vec![ligatures(&[(1, &[(&[5], 14)])])]),
        ],
    );
    let positioning = layout_table(
        &[(b"khmr", &[0])],
        &[(b"mark", &[0])],
        vec![LookupSpec::new(
            4,
            vec![mark_attach(
                &[(8, 0, (-150, 750))],
                &[(1, &[(350, 720)]), (14, &[(500, 720)])],
            )],
        )],
    );
    let kinds = gdef(
        &[
            (1, 2, 1),
            (3, 3, 3),
            (4, 5, 1),
            (6, 6, 3),
            (7, 7, 1),
            (8, 8, 3),
            (9, 9, 1),
            (10, 10, 3),
            (11, 12, 1),
            (13, 13, 3),
            (14, 14, 2),
            (15, 15, 3),
        ],
        &[],
        &[],
    );
    font(
        &glyphs,
        &map,
        &[
            (b"GDEF", kinds),
            (b"GPOS", positioning),
            (b"GSUB", substitutions),
        ],
    )
}

/// A face for the Myanmar script, which names it `script`: ka 1, nga 2,
/// the asat 3, the virama 4, the medial ra 6, the vowel e 7, the vowel u
/// 8, the anusvara 9, the dotted circle 10, kha 11, the vowel aa 15 and
/// the visarga 16.
///
/// Nga with the asat and the virama is the kinzi 5; the virama and kha
/// are the subscript kha 12; the medial ra before ka is the wide one 13;
/// and ka with aa is one glyph, 14. The anusvara and the vowel u sit on
/// anchors over and under ka.
fn myanmar_named(script: &[u8; 4]) -> Vec<u8> {
    let mut glyphs = vec![boxed(500); 17];
    for (glyph, advance) in [
        (1, 700),
        (2, 650),
        (6, 300),
        (7, 400),
        (10, 600),
        (11, 650),
        (13, 450),
        (14, 1_000),
        (15, 350),
        (16, 300),
    ] {
        glyphs[glyph] = boxed(advance);
    }
    for glyph in [3, 4, 5, 8, 12] {
        glyphs[glyph] = mark();
    }
    // A mark the face gives room of its own.
    glyphs[9] = Glyph {
        advance: 200,
        outline: Some([20, 750, 180, 850]),
    };
    let mut map = vec![
        ('\u{1000}', 1),
        ('\u{1004}', 2),
        ('\u{103A}', 3),
        ('\u{1039}', 4),
        ('\u{103C}', 6),
        ('\u{1031}', 7),
        ('\u{102F}', 8),
        ('\u{1036}', 9),
        ('\u{25CC}', 10),
        ('\u{1001}', 11),
        ('\u{102C}', 15),
        ('\u{1038}', 16),
    ];
    map.sort_unstable();
    let substitutions = layout_table(
        &[(script, &[0, 1, 2, 3])],
        &[
            (b"blwf", &[1]),
            (b"pres", &[2]),
            (b"psts", &[3]),
            (b"rphf", &[0]),
        ],
        vec![
            LookupSpec::new(4, vec![ligatures(&[(2, &[(&[3, 4], 5)])])]),
            LookupSpec::new(4, vec![ligatures(&[(4, &[(&[11], 12)])])]),
            LookupSpec::new(4, vec![ligatures(&[(6, &[(&[1], 13)])])]),
            LookupSpec::new(4, vec![ligatures(&[(1, &[(&[15], 14)])])]),
        ],
    );
    let positioning = layout_table(
        &[(script, &[0])],
        &[(b"mark", &[0])],
        vec![LookupSpec::new(
            4,
            vec![mark_attach(
                &[(8, 0, (-150, -100)), (9, 1, (100, 750))],
                &[
                    (1, &[(350, -50), (350, 720)]),
                    (14, &[(500, -50), (500, 720)]),
                ],
            )],
        )],
    );
    let kinds = gdef(
        &[
            (1, 2, 1),
            (3, 5, 3),
            (6, 7, 1),
            (8, 9, 3),
            (10, 11, 1),
            (12, 12, 3),
            (13, 14, 2),
            (15, 16, 1),
        ],
        &[],
        &[],
    );
    font(
        &glyphs,
        &map,
        &[
            (b"GDEF", kinds),
            (b"GPOS", positioning),
            (b"GSUB", substitutions),
        ],
    )
}

/// The Myanmar face under the script's tag.
pub(crate) fn myanmar() -> Vec<u8> {
    myanmar_named(b"mym2")
}

/// The same face under the tag from before the script's shaping was set
/// down: a face that names it so is shaped with its lookups alone.
pub(crate) fn myanmar_old() -> Vec<u8> {
    myanmar_named(b"mymr")
}

/// A face for four of the scripts shaped by one shared set of rules,
/// which names them by `scripts`.
///
/// Tibetan: ka 1, the subjoined ga 2 and ra 3, the vowels i 4 and u 5,
/// the a-chung 6, the anusvara 7, the tsheg 8, and the vowel ii whole,
/// 46. Ka with the subjoined ga is one glyph, 9.
///
/// Sinhala: ka 10, the al-lakuna 11, ra 12, ya 13, the kombuva 14, the
/// aela-pilla 15, the anusvara 16, the joiner 17, the vowel o whole, 40,
/// and the letter a, 45. The al-lakuna, the joiner and ra are the
/// rakaransaya 18; ra, the al-lakuna and the joiner are the repaya 19;
/// the al-lakuna, the joiner and ya are the yansaya 20.
///
/// Javanese: ka 21, the pangkon 22, ta 23, the taling 24, the wulu 25,
/// the cecak 26, the layar 27, the cakra 28, the dotted circle 30, the
/// no-break space 31 and the space 32. The pangkon and ta are the ta
/// written below, 29; the cakra is given its form before the letter, 33;
/// ka is 38 alone, 35 first, 36 between and 37 last in a row of
/// clusters; the taling before a ka alone is one glyph with it, 34; and
/// the wulu with the cecak is one mark, 39.
///
/// Balinese: the letter a 41, the tedung 42, the two as one letter 43,
/// and the ulu 44.
///
/// The marks sit on anchors under and over each script's ka, and over
/// the Balinese letters.
fn universal_named(scripts: &[&[u8; 4]]) -> Vec<u8> {
    let mut glyphs = vec![boxed(500); 47];
    for (glyph, advance) in [
        (1, 700),
        (8, 200),
        (9, 700),
        (10, 800),
        (12, 600),
        (13, 650),
        (14, 450),
        (15, 350),
        (16, 300),
        (20, 400),
        (21, 900),
        (22, 300),
        (23, 850),
        (24, 500),
        (30, 600),
        (31, 250),
        (32, 250),
        (33, 350),
        (34, 1_300),
        (35, 910),
        (36, 920),
        (37, 930),
        (38, 940),
        (40, 1_200),
        (41, 1_100),
        (42, 450),
        (43, 1_400),
        (45, 750),
    ] {
        glyphs[glyph] = boxed(advance);
    }
    for glyph in [2, 3, 4, 5, 6, 7, 11, 18, 19, 25, 26, 27, 28, 29, 39, 44, 46] {
        glyphs[glyph] = mark();
    }
    // The joiner draws nothing.
    glyphs[17] = Glyph {
        advance: 0,
        outline: None,
    };
    let mut map = vec![
        ('\u{F40}', 1),
        ('\u{F92}', 2),
        ('\u{FB2}', 3),
        ('\u{F72}', 4),
        ('\u{F74}', 5),
        ('\u{F71}', 6),
        ('\u{F7E}', 7),
        ('\u{F0B}', 8),
        ('\u{F73}', 46),
        ('\u{D9A}', 10),
        ('\u{DCA}', 11),
        ('\u{DBB}', 12),
        ('\u{DBA}', 13),
        ('\u{DD9}', 14),
        ('\u{DCF}', 15),
        ('\u{D82}', 16),
        ('\u{200D}', 17),
        ('\u{DDC}', 40),
        ('\u{D85}', 45),
        ('\u{A98F}', 21),
        ('\u{A9C0}', 22),
        ('\u{A9A0}', 23),
        ('\u{A9BA}', 24),
        ('\u{A9B6}', 25),
        ('\u{A981}', 26),
        ('\u{A982}', 27),
        ('\u{A9BF}', 28),
        ('\u{25CC}', 30),
        ('\u{A0}', 31),
        (' ', 32),
        ('\u{1B05}', 41),
        ('\u{1B35}', 42),
        ('\u{1B06}', 43),
        ('\u{1B36}', 44),
    ];
    map.sort_unstable();
    let every: Vec<u16> = (0..12).collect();
    let listed: Vec<(&[u8; 4], &[u16])> = scripts
        .iter()
        .map(|script| (*script, every.as_slice()))
        .collect();
    let substitutions = layout_table(
        &listed,
        &[
            (b"abvs", &[10]),
            (b"blwf", &[3]),
            (b"blws", &[11]),
            (b"fina", &[9]),
            (b"init", &[7]),
            (b"isol", &[6]),
            (b"medi", &[8]),
            (b"pref", &[4]),
            (b"pres", &[5]),
            (b"pstf", &[2]),
            (b"rkrf", &[1]),
            (b"rphf", &[0]),
        ],
        vec![
            LookupSpec::new(4, vec![ligatures(&[(12, &[(&[11, 17], 19)])])]),
            LookupSpec::new(4, vec![ligatures(&[(11, &[(&[17, 12], 18)])])]),
            LookupSpec::new(4, vec![ligatures(&[(11, &[(&[17, 13], 20)])])]),
            LookupSpec::new(4, vec![ligatures(&[(22, &[(&[23], 29)])])]),
            LookupSpec::new(1, vec![single(&[(28, 33)])]),
            LookupSpec::new(4, vec![ligatures(&[(24, &[(&[38], 34)])])]),
            LookupSpec::new(1, vec![single(&[(21, 38)])]),
            LookupSpec::new(1, vec![single(&[(21, 35)])]),
            LookupSpec::new(1, vec![single(&[(21, 36)])]),
            LookupSpec::new(1, vec![single(&[(21, 37)])]),
            LookupSpec::new(4, vec![ligatures(&[(25, &[(&[26], 39)])])]),
            LookupSpec::new(4, vec![ligatures(&[(1, &[(&[2], 9)])])]),
        ],
    );
    let under_over: &[(i16, i16)] = &[(450, -60), (450, 760)];
    let placed: Vec<(&[u8; 4], &[u16])> = scripts
        .iter()
        .map(|script| (*script, [0_u16].as_slice()))
        .collect();
    let positioning = layout_table(
        &placed,
        &[(b"mark", &[0])],
        vec![LookupSpec::new(
            4,
            vec![mark_attach(
                &[
                    (3, 0, (-150, -100)),
                    (4, 0, (-150, -100)),
                    (5, 1, (-150, 750)),
                    (7, 1, (-150, 750)),
                    (18, 0, (-200, -100)),
                    (19, 1, (-100, 800)),
                    (25, 1, (-150, 750)),
                    (26, 1, (-150, 750)),
                    (27, 1, (-150, 750)),
                    (28, 0, (-150, -100)),
                    (29, 0, (-200, -150)),
                    (39, 1, (-150, 750)),
                    (44, 1, (-150, 750)),
                ],
                &[
                    (1, &[(350, -50), (350, 720)]),
                    (9, &[(350, -250), (350, 720)]),
                    (10, &[(400, -50), (400, 780)]),
                    (21, under_over),
                    (34, &[(850, -60), (850, 760)]),
                    (35, under_over),
                    (36, under_over),
                    (37, under_over),
                    (38, under_over),
                    (41, &[(550, -60), (550, 800)]),
                    (43, &[(700, -60), (700, 800)]),
                ],
            )],
        )],
    );
    let kinds = gdef(
        &[
            (1, 1, 1),
            (2, 7, 3),
            (8, 8, 1),
            (9, 9, 2),
            (10, 10, 1),
            (11, 11, 3),
            (12, 14, 1),
            (15, 15, 3),
            (16, 16, 1),
            (18, 19, 3),
            (20, 24, 1),
            (25, 29, 3),
            (30, 33, 1),
            (34, 34, 2),
            (35, 38, 1),
            (39, 39, 3),
            (40, 43, 1),
            (44, 44, 3),
            (45, 45, 1),
            (46, 46, 3),
        ],
        &[],
        &[],
    );
    font(
        &glyphs,
        &map,
        &[
            (b"GDEF", kinds),
            (b"GPOS", positioning),
            (b"GSUB", substitutions),
        ],
    )
}

/// The face for the scripts shaped by the shared rules, under their own
/// names.
pub(crate) fn universal() -> Vec<u8> {
    universal_named(&[b"bali", b"java", b"sinh", b"tibt"])
}

/// The same face with only a default script: one that names no script
/// of its own was drawn to be shaped with its lookups alone.
pub(crate) fn universal_plain() -> Vec<u8> {
    universal_named(&[b"DFLT"])
}

/// A face for two scripts that join their letters as Arabic does and
/// are cut into clusters besides, which names them by `scripts`.
///
/// Mongolian: the letter a 1, and 2 first, 3 between and 4 last in a
/// word; the first variation selector 5, which draws nothing but is 8
/// after a first letter and 7 after a last one; the mark dagalga 9.
///
/// N'Ko: the letter a 11, and 12 first, 13 between and 14 last; the
/// high tone 15, the nasal mark 16, and the joiner of marks 17, which
/// draws nothing. The nasal mark right after the tone is 18, and the
/// tone right after the nasal mark 19.
///
/// The dotted circle is 10 and the space 21. The marks sit on anchors
/// over and under each letter and the circle.
fn joined_named(scripts: &[&[u8; 4]]) -> Vec<u8> {
    let mut glyphs = vec![boxed(500); 22];
    for (glyph, advance) in [
        (1, 600),
        (2, 610),
        (3, 620),
        (4, 630),
        (7, 300),
        (8, 310),
        (10, 600),
        (11, 500),
        (12, 510),
        (13, 520),
        (14, 530),
        (21, 250),
    ] {
        glyphs[glyph] = boxed(advance);
    }
    for glyph in [9, 15, 16, 18, 19] {
        glyphs[glyph] = mark();
    }
    for glyph in [5, 17] {
        glyphs[glyph] = Glyph {
            advance: 0,
            outline: None,
        };
    }
    let mut map = vec![
        ('\u{1820}', 1),
        ('\u{180B}', 5),
        ('\u{18A9}', 9),
        ('\u{25CC}', 10),
        ('\u{7CA}', 11),
        ('\u{7EB}', 15),
        ('\u{7F2}', 16),
        ('\u{34F}', 17),
        (' ', 21),
    ];
    map.sort_unstable();
    let every: Vec<u16> = (0..4).collect();
    let listed: Vec<(&[u8; 4], &[u16])> = scripts
        .iter()
        .map(|script| (*script, every.as_slice()))
        .collect();
    let substitutions = layout_table(
        &listed,
        &[
            (b"calt", &[5, 6]),
            (b"fina", &[2]),
            (b"init", &[0]),
            (b"medi", &[1]),
        ],
        vec![
            LookupSpec::new(1, vec![single(&[(1, 2), (5, 8), (11, 12)])]),
            LookupSpec::new(1, vec![single(&[(1, 3), (11, 13)])]),
            LookupSpec::new(1, vec![single(&[(1, 4), (5, 7), (11, 14)])]),
            LookupSpec::new(1, vec![single(&[(16, 18)])]),
            LookupSpec::new(1, vec![single(&[(15, 19)])]),
            LookupSpec::new(6, vec![chain(&[&[15]], &[&[16]], &[], &[(0, 3)])]),
            LookupSpec::new(6, vec![chain(&[&[16]], &[&[15]], &[], &[(0, 4)])]),
        ],
    );
    let over_under: &[(i16, i16)] = &[(300, 720), (300, -50)];
    let placed: Vec<(&[u8; 4], &[u16])> = scripts
        .iter()
        .map(|script| (*script, [0_u16].as_slice()))
        .collect();
    let positioning = layout_table(
        &placed,
        &[(b"mark", &[0])],
        vec![LookupSpec::new(
            4,
            vec![mark_attach(
                &[
                    (9, 0, (-150, 750)),
                    (15, 0, (-150, 750)),
                    (16, 1, (-150, -100)),
                    (18, 1, (-150, -100)),
                    (19, 0, (-150, 750)),
                ],
                &[
                    (1, over_under),
                    (2, over_under),
                    (3, over_under),
                    (4, over_under),
                    (10, over_under),
                    (11, over_under),
                    (12, over_under),
                    (13, over_under),
                    (14, over_under),
                ],
            )],
        )],
    );
    let kinds = gdef(
        &[
            (1, 4, 1),
            (7, 8, 1),
            (9, 9, 3),
            (10, 14, 1),
            (15, 16, 3),
            (18, 19, 3),
            (21, 21, 1),
        ],
        &[],
        &[],
    );
    font(
        &glyphs,
        &map,
        &[
            (b"GDEF", kinds),
            (b"GPOS", positioning),
            (b"GSUB", substitutions),
        ],
    )
}

/// The face for the joined scripts, under their own names.
pub(crate) fn joined() -> Vec<u8> {
    joined_named(&[b"mong", b"nko "])
}

/// The same face with only a default script: its letters are then not
/// joined at all.
pub(crate) fn joined_plain() -> Vec<u8> {
    joined_named(&[b"DFLT"])
}

/// A fixture's file name and what builds it.
type Built = (&'static str, fn() -> Vec<u8>);

const FILES: [Built; 15] = [
    ("shaping.ttf", shaping),
    ("hostile.ttf", hostile),
    ("machines.ttf", machines),
    ("varied.ttf", varied),
    ("han.ttf", han),
    ("han-kana.ttf", han_kana),
    ("han-hangul.ttf", han_hangul),
    ("khmer.ttf", khmer),
    ("myanmar.ttf", myanmar),
    ("myanmar-old.ttf", myanmar_old),
    ("universal.ttf", universal),
    ("universal-plain.ttf", universal_plain),
    ("hostile-syllables.ttf", hostile_syllables),
    ("joined.ttf", joined),
    ("joined-plain.ttf", joined_plain),
];

fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

#[test]
fn the_fixture_files_are_what_the_builder_writes() {
    for (name, build) in FILES {
        let stored = std::fs::read(path(name)).expect("fixture");
        assert!(
            stored == build(),
            "{name} is stale: write the fixtures again"
        );
        crate::read(&stored).expect("font");
    }
}

#[test]
#[ignore = "writes the fixture fonts"]
fn write_the_fixture_fonts() {
    for (name, build) in FILES {
        std::fs::write(path(name), build()).expect("write");
    }
}

#[test]
fn the_machines_face_reads_its_chain_and_its_states() {
    use crate::morx::{CLASS_DELETED, CLASS_END, CLASS_OTHER, DELETED_GLYPH, Metamorphosis};
    let bytes = machines();
    let face = crate::read(&bytes).expect("font");
    let chains = face.morx().expect("morx").chains().expect("chains");
    assert_eq!(chains.len(), 1);
    assert_eq!(chains[0].default_flags, 1);
    assert_eq!(chains[0].subtables.len(), 2);
    let Metamorphosis::Swap(swap) = chains[0].subtables[0].kind else {
        panic!("swap");
    };
    assert_eq!(
        (swap.value(5), swap.value(2), swap.value(9)),
        (Some(3), None, None)
    );
    let Metamorphosis::Ligature {
        machine,
        actions,
        components,
        ligatures,
    } = chains[0].subtables[1].kind
    else {
        panic!("ligature");
    };
    assert_eq!((machine.class(2), machine.class(3)), (4, 5));
    assert_eq!(machine.class(5), CLASS_OTHER);
    assert_eq!(machine.class(DELETED_GLYPH), CLASS_DELETED);
    // An A from the start goes to state 2 and is marked; a B there joins.
    assert_eq!(machine.entry(0, 4).expect("entry"), [0, 2, 0x80, 0, 0, 0]);
    assert_eq!(machine.entry(2, 5).expect("entry"), [0, 0, 0xA0, 0, 0, 0]);
    assert_eq!(machine.entry(2, CLASS_END).expect("entry"), [0; 6]);
    assert!(machine.entry(0, 6).is_err());
    assert!(machine.entry(9, 0).is_err());
    assert_eq!(
        crate::morx::long_at(actions, 1).expect("action"),
        0xFFFF_FFFF
    );
    assert_eq!(crate::morx::word_at(components, 1).expect("component"), 0);
    assert_eq!(crate::morx::word_at(ligatures, 0).expect("ligature"), 4);
    assert!(crate::morx::word_at(ligatures, 1).is_err());

    // A torn table is an error; a version this reader does not know is
    // no table.
    let count = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
    let record = (0..count)
        .map(|slot| 12 + slot * 16)
        .find(|at| &bytes[*at..*at + 4] == b"morx")
        .expect("morx");
    let at = u32::from_be_bytes([
        bytes[record + 8],
        bytes[record + 9],
        bytes[record + 10],
        bytes[record + 11],
    ]) as usize;
    let mut newer = bytes.clone();
    newer[at + 1] = 9;
    assert!(crate::read(&newer).expect("font").morx().is_none());
    let mut torn = bytes.clone();
    torn[at + 7] = 200;
    let chains = crate::read(&torn)
        .expect("font")
        .morx()
        .expect("morx")
        .chains();
    assert!(chains.is_err());
}

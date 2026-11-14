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

/// A face whose rules never settle: eight lookups that each write four
/// glyphs for one, and two contextual lookups that call each other.
pub(crate) fn hostile() -> Vec<u8> {
    let glyphs = vec![boxed(500); 4];
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
    let substitutions = layout_table(
        &[(b"DFLT", &[0, 1])],
        &[(b"calt", &[8]), (b"ccmp", &[0, 1, 2, 3, 4, 5, 6, 7])],
        lookups,
    );
    font(&glyphs, &[('A', 2), ('B', 3)], &[(b"GSUB", substitutions)])
}

/// A fixture's file name and what builds it.
type Built = (&'static str, fn() -> Vec<u8>);

const FILES: [Built; 2] = [("shaping.ttf", shaping), ("hostile.ttf", hostile)];

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
        let face = crate::read(&stored).expect("font");
        assert!(face.gsub().is_some(), "{name}");
    }
}

#[test]
#[ignore = "writes the fixture fonts"]
fn write_the_fixture_fonts() {
    for (name, build) in FILES {
        std::fs::write(path(name), build()).expect("write");
    }
}

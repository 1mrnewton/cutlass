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

/// A fixture's file name and what builds it.
type Built = (&'static str, fn() -> Vec<u8>);

const FILES: [Built; 4] = [
    ("shaping.ttf", shaping),
    ("hostile.ttf", hostile),
    ("machines.ttf", machines),
    ("varied.ttf", varied),
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
        let face = crate::read(&stored).expect("font");
        let varied = face.variant('1', '\u{FE0F}') != Ok(crate::Variant::Unlisted);
        assert!(
            face.gsub().is_some() || face.morx().is_some() || varied,
            "{name}"
        );
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

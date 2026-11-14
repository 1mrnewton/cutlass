//! Fixture faces that kern as faces made for Apple's layout engine do:
//! one with a version 1 `kern` table and one with `kerx` and `ankr`.
//!
//! To write them again after a change here:
//! `cargo test -p cutlass-font -- --ignored write_the_kern_fixture_fonts`.

use crate::fixture::{Glyph, be, boxed, font};

/// The glyphs of both faces: `A` 1, `B` 2, a combining acute 3 and `C` 4,
/// each letter 600 wide.
const A: u16 = 1;
const B: u16 = 2;
const ACUTE: u16 = 3;
const C: u16 = 4;

fn glyphs() -> Vec<Glyph> {
    vec![
        boxed(500),
        boxed(600),
        boxed(600),
        Glyph {
            advance: 0,
            outline: Some([-200, 750, -100, 850]),
        },
        boxed(600),
    ]
}

const MAP: [(char, u16); 4] = [('A', A), ('B', B), ('C', C), ('\u{301}', ACUTE)];

fn signed(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

fn long(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

/// One step of a machine: the state after it, its flags, and the values
/// or the action it names.
type Step = (u16, u16, u16);

/// A state machine as the `kern` table stores one, for its format 1: a
/// class for `first` and for the glyph after it (4 and 5), `rows` of an
/// entry number for each of six classes, the `steps` those number, and
/// the `values` the steps name by their place in the list.
fn byte_machine(first: u16, rows: &[[u8; 6]], steps: &[Step], values: &[i16]) -> Vec<u8> {
    const CLASSES: usize = 6;
    let class_table = 10;
    let states = class_table + 6;
    let entries = states + rows.len() * CLASSES;
    let value_table = entries + steps.len() * 4;
    let mut body = be(&[
        6,
        u16::try_from(class_table).expect("offset"),
        u16::try_from(states).expect("offset"),
        u16::try_from(entries).expect("offset"),
        u16::try_from(value_table).expect("offset"),
    ]);
    body.extend(be(&[first, 2]));
    body.extend([4, 5]);
    for row in rows {
        body.extend(row);
    }
    for (next, flags, value) in steps {
        // The next state is the byte offset of its row, and a step that
        // moves glyphs names its first value by its byte offset.
        let next = states + usize::from(*next) * CLASSES;
        let named = if *value == u16::MAX {
            0
        } else {
            u16::try_from(value_table + usize::from(*value) * 2).expect("value")
        };
        body.extend(be(&[u16::try_from(next).expect("state"), flags | named]));
    }
    body.extend(signed(values));
    body
}

/// A subtable of a version 1 `kern` table.
fn kern_subtable(coverage: u16, body: &[u8]) -> Vec<u8> {
    let mut bytes = long(&[u32::try_from(body.len() + 8).expect("length")]);
    bytes.extend(be(&[coverage, 0]));
    bytes.extend(body);
    bytes
}

const PUSH: u16 = 0x8000;
const NONE: u16 = u16::MAX;

/// A face with a version 1 `kern` table of four passes.
///
/// A machine along the line: `B` straight after `A` moves back 100 and
/// `A` 50. A machine across the line: an acute straight after `B` is
/// lifted 300, and the glyph after the acute is set back on the line.
/// Classes by offset: `A` after `C` is 80 closer. Compact classes: `C`
/// after `A` is 40 closer.
pub(crate) fn kerned() -> Vec<u8> {
    // Classes: 4 is A, 5 is B. State 2 has seen an A.
    let along = byte_machine(
        A,
        &[[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 2]],
        &[(0, 0, NONE), (2, PUSH, NONE), (0, PUSH, 0)],
        // B first, then A, whose odd value ends the list.
        &[-100, -49],
    );
    // Classes: 4 is B, 5 is the acute. State 2 has seen a B, and state 3
    // an acute after it: whatever comes next goes back on the line.
    let across = byte_machine(
        B,
        &[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 2],
            [3, 3, 3, 3, 4, 3],
        ],
        &[
            (0, 0, NONE),
            (2, 0, NONE),
            (3, PUSH, 0),
            (0, PUSH, 1),
            (2, PUSH, 1),
        ],
        &[301, -0x7FFF],
    );
    // Format 2: the left class of C is the offset of its row, the right
    // class of A the offset of its column, and the array one row of two.
    let array = 8 + 8 + 6 + 6;
    let mut offsets = be(&[4, 16, 22, array]);
    offsets.extend(be(&[C, 1, array]));
    offsets.extend(be(&[A, 1, 2]));
    offsets.extend(signed(&[0, -80]));
    // Format 3: five glyphs, two values, two classes a side.
    let mut compact = be(&[5]);
    compact.extend([2, 2, 2, 0]);
    compact.extend(signed(&[0, -40]));
    compact.extend([0, 1, 0, 0, 0]);
    compact.extend([0, 0, 0, 0, 1]);
    compact.extend([0, 0, 0, 1]);
    let mut kern = long(&[0x0001_0000, 4]);
    kern.extend(kern_subtable(1, &along));
    kern.extend(kern_subtable(0x4001, &across));
    kern.extend(kern_subtable(2, &offsets));
    kern.extend(kern_subtable(3, &compact));
    font(&glyphs(), &MAP, &[(b"kern", kern)])
}

/// A face whose one kerning machine never moves on: every step asks to
/// read the same glyph again, and remembers it, and moves what it
/// remembers back by 2.
pub(crate) fn stuck() -> Vec<u8> {
    const STAY: u16 = 0x4000;
    let machine = byte_machine(A, &[[0; 6], [0; 6]], &[(0, PUSH | STAY, 0)], &[-1]);
    let mut kern = long(&[0x0001_0000, 1]);
    kern.extend(kern_subtable(1, &machine));
    font(&glyphs(), &MAP, &[(b"kern", kern)])
}

/// A lookup from single glyphs to values, as the extended tables store
/// one: format 6, in glyph order.
fn lookup(pairs: &[(u16, u16)]) -> Vec<u8> {
    let mut table = be(&[6, 4, u16::try_from(pairs.len()).expect("count"), 0, 0, 0]);
    for (glyph, value) in pairs {
        table.extend(be(&[*glyph, *value]));
    }
    table
}

/// A state machine as `kerx` stores one: a lookup of `classes`, `rows`
/// of an entry number for each of six classes, the `steps`, and after
/// its four offsets a fifth field: where `tail` is, with `flags` on top.
fn word_machine(
    classes: &[(u16, u16)],
    rows: &[[u16; 6]],
    steps: &[Step],
    flags: u32,
    tail: &[u8],
) -> Vec<u8> {
    let classes = lookup(classes);
    let class_table = 20;
    let states = class_table + classes.len();
    let entries = states + rows.len() * 12;
    let tail_at = entries + steps.len() * 6;
    let place = |value: usize| u32::try_from(value).expect("offset");
    let mut body = long(&[
        6,
        place(class_table),
        place(states),
        place(entries),
        flags | place(tail_at),
    ]);
    body.extend(classes);
    for row in rows {
        body.extend(be(row));
    }
    for (next, flags, value) in steps {
        body.extend(be(&[*next, *flags, *value]));
    }
    body.extend(tail);
    body
}

/// A subtable of a `kerx` table.
fn kerx_subtable(coverage: u32, body: &[u8]) -> Vec<u8> {
    let mut bytes = long(&[u32::try_from(body.len() + 12).expect("length"), coverage, 0]);
    bytes.extend(body);
    bytes
}

/// The `ankr` table: `A` has one anchor at (300, 700) and the acute one
/// at (-150, 600).
fn anchors() -> Vec<u8> {
    let lists = {
        let mut lists = long(&[1]);
        lists.extend(signed(&[300, 700]));
        lists.extend(long(&[1]));
        lists.extend(signed(&[-150, 600]));
        lists
    };
    let lookup = lookup(&[(A, 0), (ACUTE, 8)]);
    let mut table = be(&[0, 0]);
    table.extend(long(&[
        12,
        u32::try_from(12 + lookup.len()).expect("offset"),
    ]));
    table.extend(lookup);
    table.extend(lists);
    table
}

const MARK: u16 = 0x8000;

/// A face with a `kerx` table of four passes and the `ankr` table one of
/// them reads.
///
/// Sorted pairs: `B` after `A` is 60 closer. Classes by lookup: `A`
/// after `C` is 90 closer. A machine: `C` straight after `B` moves back
/// 70. An attaching machine: an acute after `A` is set with its anchor on
/// `A`'s.
pub(crate) fn extended() -> Vec<u8> {
    let mut sorted = long(&[1, 0, 0, 0]);
    sorted.extend(be(&[A, B]));
    sorted.extend(signed(&[-60]));
    // Two rows of two values: the left value of C is the byte offset of
    // the second row, and the right value of A that of the second column.
    let left = lookup(&[(C, 4)]);
    let right = lookup(&[(A, 2)]);
    let left_at = 12 + 16;
    let right_at = left_at + left.len();
    let array_at = right_at + right.len();
    let place = |value: usize| u32::try_from(value).expect("offset");
    let mut classes = long(&[4, place(left_at), place(right_at), place(array_at)]);
    classes.extend(left);
    classes.extend(right);
    classes.extend(signed(&[0, 0, 0, -90]));
    // Classes: 4 is B, 5 is C. State 2 has seen a B.
    let along = word_machine(
        &[(B, 4), (C, 5)],
        &[[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 2]],
        &[(0, 0, NONE), (2, 0, NONE), (0, PUSH, 0)],
        0,
        &signed(&[-69]),
    );
    // Classes: 4 is A, 5 is the acute. State 2 remembers an A.
    let attach = word_machine(
        &[(A, 4), (ACUTE, 5)],
        &[[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 2]],
        &[(0, 0, NONE), (2, MARK, NONE), (2, 0, 0)],
        // Anchors by their number in the `ankr` table.
        1 << 30,
        &be(&[0, 0]),
    );
    let mut kerx = be(&[2, 0]);
    kerx.extend(long(&[4]));
    kerx.extend(kerx_subtable(0, &sorted));
    kerx.extend(kerx_subtable(2, &classes));
    kerx.extend(kerx_subtable(1, &along));
    kerx.extend(kerx_subtable(4, &attach));
    font(&glyphs(), &MAP, &[(b"kerx", kerx), (b"ankr", anchors())])
}

/// A face with glyphs far wider than an em, as a phrase drawn as one
/// glyph is: `A` is six ems across and one tall, `B` twenty ems across,
/// and `C` thirty-two, which is past any grid a glyph is painted on.
pub(crate) fn wide() -> Vec<u8> {
    let across = |ems: i16| Glyph {
        advance: u16::try_from(ems).expect("ems") * 1_000,
        outline: Some([0, 0, ems * 1_000, 1_000]),
    };
    let glyphs = [
        boxed(500),
        across(6),
        across(20),
        Glyph {
            advance: 500,
            outline: Some([-16_000, 0, 16_000, 1_000]),
        },
    ];
    font(&glyphs, &[('A', 1), ('B', 2), ('C', 3)], &[])
}

/// A fixture's file name and what builds it.
type Built = (&'static str, fn() -> Vec<u8>);

const FILES: [Built; 3] = [
    ("kern-machine.ttf", kerned),
    ("kerx.ttf", extended),
    ("kern-stuck.ttf", stuck),
];

/// Fixtures with no kerning, kept and checked the same way.
const OTHER_FILES: [Built; 1] = [("wide.ttf", wide)];

fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

#[test]
fn the_kern_fixture_files_are_what_the_builder_writes() {
    for (name, build) in FILES {
        let stored = std::fs::read(path(name)).expect("fixture");
        assert!(
            stored == build(),
            "{name} is stale: write the fixtures again"
        );
        let face = crate::read(&stored).expect("font");
        assert!(!face.kern_passes().is_empty() || !face.kerx_passes().is_empty());
    }
    for (name, build) in OTHER_FILES {
        let stored = std::fs::read(path(name)).expect("fixture");
        assert!(
            stored == build(),
            "{name} is stale: write the fixtures again"
        );
        crate::read(&stored).expect("font");
    }
}

#[test]
#[ignore = "writes the kern fixture fonts"]
fn write_the_kern_fixture_fonts() {
    for (name, build) in FILES.iter().chain(&OTHER_FILES) {
        std::fs::write(path(name), build()).expect("write");
    }
}
